#!/usr/bin/env python3
"""
Simplified RAG system to avoid memory issues
"""

import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

@dataclass
class SimpleChunk:
    """Simple conversation chunk"""
    chunk_id: str
    text: str
    date: str
    timestamp: int

class SimpleRAG:
    """Lightweight RAG system for conversation Q&A"""
    
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.normalized = phone_number.replace('(', '').replace(')', '').replace(' ', '').replace('-', '')
        
        # Initialize embedding model
        console.print("🧠 Loading lightweight embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Paths
        self.conv_file = f"data/conversation_{self.normalized}.json"
        self.db_file = f"data/simple_rag_{self.normalized}.db"
        
        # Load data
        self.messages = []
        self.chunks = []
        
        self._load_conversation()
        self._setup_database()
        self._create_chunks()
    
    def _load_conversation(self):
        """Load conversation messages"""
        if not Path(self.conv_file).exists():
            console.print(f"❌ Conversation file not found: {self.conv_file}")
            return
            
        with open(self.conv_file, 'r') as f:
            self.messages = json.load(f)
        
        console.print(f"📱 Loaded {len(self.messages)} messages")
    
    def _setup_database(self):
        """Setup SQLite database for text storage"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_chunks(timestamp);
        """)
        
        # Enable FTS for text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id, text, date
            );
        """)
        
        conn.commit()
        conn.close()
    
    def _create_chunks(self):
        """Create conversation chunks"""
        # Check if chunks already exist
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversation_chunks")
        existing_count = cursor.fetchone()[0]
        conn.close()
        
        if existing_count > 0:
            console.print(f"📚 Using existing {existing_count} chunks")
            return
        
        console.print("🔨 Creating conversation chunks...")
        
        chunk_size = 10  # Larger chunks for better context
        chunks = []
        
        for i in range(0, len(self.messages), chunk_size):
            end_idx = min(i + chunk_size, len(self.messages))
            chunk_messages = self.messages[i:end_idx]
            
            if not chunk_messages:
                continue
            
            # Create chunk text
            chunk_parts = []
            for msg in chunk_messages:
                sender = "You" if msg['is_from_me'] else "Them"
                chunk_parts.append(f"{sender}: {msg['text']}")
            
            chunk_text = "\n".join(chunk_parts)
            
            chunk = SimpleChunk(
                chunk_id=f"chunk_{i}_{end_idx}",
                text=chunk_text,
                date=chunk_messages[0]['date'],
                timestamp=chunk_messages[0]['timestamp']
            )
            
            chunks.append(chunk)
        
        # Store in database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO conversation_chunks 
                (chunk_id, text, date, timestamp)
                VALUES (?, ?, ?, ?)
            """, (chunk.chunk_id, chunk.text, chunk.date, chunk.timestamp))
            
            # Add to FTS
            cursor.execute("""
                INSERT INTO chunks_fts (chunk_id, text, date)
                VALUES (?, ?, ?)
            """, (chunk.chunk_id, chunk.text, chunk.date))
        
        conn.commit()
        conn.close()
        
        console.print(f"✅ Created {len(chunks)} chunks")
    
    def search_text(self, query: str, top_k: int = 5) -> List[SimpleChunk]:
        """Search using SQLite FTS"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            # Clean query for FTS - escape special characters and use simple terms
            clean_query = ' '.join(query.split())  # Remove extra spaces
            clean_query = clean_query.replace('"', '').replace("'", "")  # Remove quotes
            
            # FTS search with proper escaping
            cursor.execute("""
                SELECT c.chunk_id, c.text, c.date, c.timestamp
                FROM chunks_fts 
                JOIN conversation_chunks c ON chunks_fts.chunk_id = c.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """, (f'"{clean_query}"', top_k))
            
            results = []
            for row in cursor.fetchall():
                chunk = SimpleChunk(
                    chunk_id=row[0],
                    text=row[1],
                    date=row[2],
                    timestamp=row[3]
                )
                results.append(chunk)
            
            # If no results with exact phrase, try individual words
            if not results and len(clean_query.split()) > 1:
                words = clean_query.split()[:3]  # Limit to first 3 words
                word_query = ' OR '.join(words)
                
                cursor.execute("""
                    SELECT c.chunk_id, c.text, c.date, c.timestamp
                    FROM chunks_fts 
                    JOIN conversation_chunks c ON chunks_fts.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                """, (word_query, top_k))
                
                for row in cursor.fetchall():
                    chunk = SimpleChunk(
                        chunk_id=row[0],
                        text=row[1],
                        date=row[2],
                        timestamp=row[3]
                    )
                    results.append(chunk)
            
        except sqlite3.OperationalError as e:
            # Fallback to LIKE search if FTS fails
            console.print(f"⚠️  FTS search failed, using LIKE search: {e}")
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp
                FROM conversation_chunks
                WHERE text LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f'%{query}%', top_k))
            
            results = []
            for row in cursor.fetchall():
                chunk = SimpleChunk(
                    chunk_id=row[0],
                    text=row[1],
                    date=row[2],
                    timestamp=row[3]
                )
                results.append(chunk)
        
        conn.close()
        return results
    
    def _get_relationship_context(self, question: str) -> List[SimpleChunk]:
        """Get broader conversation context for relationship questions"""
        # Search for relationship indicators in conversations
        relationship_terms = [
            'cute', 'sweet', 'miss you', 'love', 'heart', 'date', 'dinner',
            'hang out', 'see you', 'meet up', 'together', 'us', 'we should',
            'want to', 'would you', 'do you want', 'lets', "let's",
            'flirt', 'tease', 'compliment', 'beautiful', 'handsome',
            'feelings', 'like you', 'crush', 'attracted', 'chemistry',
            'kiss', 'hug', 'hold hands', 'romantic', 'special'
        ]
        
        # Get recent conversations (more context for relationship questions)
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get chunks from different time periods for broader context
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        
        recent_chunks = []
        for row in cursor.fetchall():
            chunk = SimpleChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3]
            )
            recent_chunks.append(chunk)
        
        # Also search for chunks containing relationship indicators
        relationship_chunks = []
        for term in relationship_terms[:10]:  # Limit to avoid too many queries
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp
                FROM conversation_chunks
                WHERE text LIKE ?
                ORDER BY timestamp DESC
                LIMIT 3
            """, (f'%{term}%',))
            
            for row in cursor.fetchall():
                chunk = SimpleChunk(
                    chunk_id=row[0],
                    text=row[1],
                    date=row[2],
                    timestamp=row[3]
                )
                if chunk not in relationship_chunks:
                    relationship_chunks.append(chunk)
        
        conn.close()
        
        # Combine and deduplicate
        all_chunks = recent_chunks + relationship_chunks
        seen_ids = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        # Return top 8 for better context
        return unique_chunks[:8]
    
    def _get_timeline_context(self, question: str) -> List[SimpleChunk]:
        """Get chronological context for timeline questions"""
        from datetime import datetime, timedelta
        import calendar
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get all chunks sorted by time
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            ORDER BY timestamp ASC
        """)
        
        all_chunks = []
        for row in cursor.fetchall():
            chunk = SimpleChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3]
            )
            all_chunks.append(chunk)
        
        conn.close()
        
        if not all_chunks:
            return []
        
        # Create timeline samples for comprehensive analysis
        timeline_chunks = []
        
        # 1. Get very first conversations (first 5 chunks)
        timeline_chunks.extend(all_chunks[:5])
        
        # 2. Get samples from different time periods
        total_chunks = len(all_chunks)
        
        if total_chunks > 100:
            # Sample from different quarters of the conversation
            quarter_size = total_chunks // 4
            
            # Early period (first quarter)
            timeline_chunks.extend(all_chunks[quarter_size//2:quarter_size//2 + 3])
            
            # Mid-early period (second quarter)  
            start_idx = quarter_size + quarter_size//2
            timeline_chunks.extend(all_chunks[start_idx:start_idx + 3])
            
            # Mid-late period (third quarter)
            start_idx = 2 * quarter_size + quarter_size//2
            timeline_chunks.extend(all_chunks[start_idx:start_idx + 3])
            
            # Recent period (last quarter)
            timeline_chunks.extend(all_chunks[-10:])
        else:
            # For smaller conversations, sample evenly
            step = max(1, total_chunks // 15)
            timeline_chunks.extend(all_chunks[::step])
        
        # 3. Add monthly samples if conversation spans months
        try:
            first_date = datetime.strptime(all_chunks[0].date.split()[0], '%Y-%m-%d')
            last_date = datetime.strptime(all_chunks[-1].date.split()[0], '%Y-%m-%d')
            
            if (last_date - first_date).days > 60:  # More than 2 months
                # Get one sample from each month
                current_date = first_date
                while current_date <= last_date:
                    month_start = current_date.replace(day=1)
                    next_month = (month_start + timedelta(days=32)).replace(day=1)
                    
                    # Find chunks from this month
                    month_chunks = [
                        chunk for chunk in all_chunks 
                        if month_start <= datetime.strptime(chunk.date.split()[0], '%Y-%m-%d') < next_month
                    ]
                    
                    if month_chunks:
                        # Take middle chunk from this month for representative sample
                        mid_idx = len(month_chunks) // 2
                        timeline_chunks.append(month_chunks[mid_idx])
                    
                    current_date = next_month
                    
        except (ValueError, IndexError):
            pass  # Date parsing failed, continue with existing chunks
        
        # Remove duplicates while preserving chronological order
        seen_ids = set()
        unique_timeline = []
        
        # Sort by timestamp to maintain chronological order
        timeline_chunks.sort(key=lambda x: x.timestamp)
        
        for chunk in timeline_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_timeline.append(chunk)
        
        # Return comprehensive timeline (up to 20 chunks for detailed analysis)
        return unique_timeline[:20]
    
    def _get_emotional_context(self, question: str) -> List[SimpleChunk]:
        """Get emotionally rich context for sentiment questions"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Emotional indicators to search for
        emotional_indicators = [
            # Positive emotions
            'love', 'happy', 'excited', 'amazing', 'wonderful', 'perfect', 'best',
            'awesome', 'great', 'fantastic', 'incredible', 'beautiful', 'sweet',
            'cute', 'adorable', 'miss you', 'care about', 'appreciate', 'grateful',
            
            # Negative emotions  
            'sad', 'upset', 'hurt', 'angry', 'frustrated', 'disappointed', 'worried',
            'anxious', 'nervous', 'scared', 'afraid', 'hate', 'annoyed', 'mad',
            
            # Relationship emotions
            'chemistry', 'connection', 'attracted', 'crush', 'feelings', 'vibe',
            'energy', 'comfortable', 'close', 'special', 'meant to be',
            
            # Expression patterns
            '😍', '❤️', '💕', '😊', '😘', '🥰', '😂', '😭', '😢', '😡', '😤',
            'lol', 'haha', 'omg', 'wow', 'ugh', 'sigh', 'yay', 'aww'
        ]
        
        emotional_chunks = []
        
        # Search for chunks with emotional content
        for indicator in emotional_indicators[:15]:  # Limit to avoid too many queries
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp
                FROM conversation_chunks
                WHERE text LIKE ?
                ORDER BY timestamp DESC
                LIMIT 3
            """, (f'%{indicator}%',))
            
            for row in cursor.fetchall():
                chunk = SimpleChunk(
                    chunk_id=row[0],
                    text=row[1],
                    date=row[2],
                    timestamp=row[3]
                )
                emotional_chunks.append(chunk)
        
        # Also get recent conversations for current emotional state
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        recent_chunks = []
        for row in cursor.fetchall():
            chunk = SimpleChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3]
            )
            recent_chunks.append(chunk)
        
        # Get conversations with longer messages (often more emotional)
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            WHERE length(text) > 200
            ORDER BY timestamp DESC
            LIMIT 8
        """)
        
        long_chunks = []
        for row in cursor.fetchall():
            chunk = SimpleChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3]
            )
            long_chunks.append(chunk)
        
        conn.close()
        
        # Combine all emotional context
        all_emotional = emotional_chunks + recent_chunks + long_chunks
        
        # Remove duplicates and sort by timestamp
        seen_ids = set()
        unique_emotional = []
        
        for chunk in all_emotional:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_emotional.append(chunk)
        
        # Sort by timestamp for chronological emotional analysis
        unique_emotional.sort(key=lambda x: x.timestamp)
        
        # Return emotional context (up to 15 chunks)
        return unique_emotional[:15]
    
    def _get_pattern_context(self, question: str) -> List[SimpleChunk]:
        """Get context with behavioral patterns and statistics"""
        from datetime import datetime, timedelta
        import re
        
        # Load all messages for pattern analysis
        all_messages = self.messages
        if not all_messages:
            return []
        
        # Calculate communication patterns
        pattern_stats = self._calculate_communication_patterns(all_messages)
        
        # Get representative conversation samples
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get samples from different time periods
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            ORDER BY timestamp ASC
        """)
        
        all_chunks = []
        for row in cursor.fetchall():
            chunk = SimpleChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3]
            )
            all_chunks.append(chunk)
        
        conn.close()
        
        if not all_chunks:
            return []
        
        # Sample conversations to show patterns
        pattern_chunks = []
        
        # Get early, middle, and recent samples
        total = len(all_chunks)
        if total > 20:
            # Early conversations
            pattern_chunks.extend(all_chunks[:3])
            # Middle conversations  
            mid_start = total // 2 - 2
            pattern_chunks.extend(all_chunks[mid_start:mid_start + 4])
            # Recent conversations
            pattern_chunks.extend(all_chunks[-5:])
        else:
            pattern_chunks = all_chunks[:12]
        
        # Create a stats chunk to include numerical data
        stats_text = f"""Communication Pattern Statistics:
        
Total Messages: {pattern_stats['total_messages']}
Your Messages: {pattern_stats['my_messages']} ({pattern_stats['my_percentage']:.1f}%)
Their Messages: {pattern_stats['their_messages']} ({pattern_stats['their_percentage']:.1f}%)

Initiative Pattern:
- Who starts conversations more: {pattern_stats['initiation_leader']}
- Your initiated conversations: {pattern_stats['my_initiations']}
- Their initiated conversations: {pattern_stats['their_initiations']}

Message Timing:
- Most active time period: {pattern_stats['most_active_period']}
- Average messages per day: {pattern_stats['avg_messages_per_day']:.1f}
- Conversation frequency: {pattern_stats['conversation_frequency']}

Message Characteristics:
- Average message length (yours): {pattern_stats['avg_my_length']:.1f} characters
- Average message length (theirs): {pattern_stats['avg_their_length']:.1f} characters
- Total conversation span: {pattern_stats['conversation_span']} days"""
        
        # Add stats as a special chunk
        stats_chunk = SimpleChunk(
            chunk_id="pattern_stats",
            text=stats_text,
            date=all_chunks[-1].date if all_chunks else "2024-01-01",
            timestamp=0
        )
        
        pattern_chunks.insert(0, stats_chunk)  # Put stats first
        
        return pattern_chunks[:15]
    
    def _calculate_communication_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """Calculate detailed communication patterns from messages"""
        from datetime import datetime
        import re
        
        if not messages:
            return {}
        
        # Basic counts
        total_messages = len(messages)
        my_messages = sum(1 for m in messages if m['is_from_me'])
        their_messages = total_messages - my_messages
        
        my_percentage = (my_messages / total_messages) * 100
        their_percentage = (their_messages / total_messages) * 100
        
        # Message lengths
        my_lengths = [len(m['text']) for m in messages if m['is_from_me']]
        their_lengths = [len(m['text']) for m in messages if not m['is_from_me']]
        
        avg_my_length = sum(my_lengths) / len(my_lengths) if my_lengths else 0
        avg_their_length = sum(their_lengths) / len(their_lengths) if their_lengths else 0
        
        # Time analysis
        try:
            first_date = datetime.strptime(messages[0]['date'].split()[0], '%Y-%m-%d')
            last_date = datetime.strptime(messages[-1]['date'].split()[0], '%Y-%m-%d')
            conversation_span = (last_date - first_date).days
            avg_messages_per_day = total_messages / max(conversation_span, 1)
        except:
            conversation_span = 1
            avg_messages_per_day = total_messages
        
        # Initiative analysis (simplified - based on time gaps)
        my_initiations = 0
        their_initiations = 0
        
        # Look for conversation starts (messages after longer gaps)
        for i in range(1, len(messages)):
            try:
                prev_time = datetime.strptime(messages[i-1]['date'], '%Y-%m-%d %H:%M:%S')
                curr_time = datetime.strptime(messages[i]['date'], '%Y-%m-%d %H:%M:%S')
                time_gap = (curr_time - prev_time).total_seconds() / 3600  # hours
                
                # If gap > 4 hours, consider this a new conversation start
                if time_gap > 4:
                    if messages[i]['is_from_me']:
                        my_initiations += 1
                    else:
                        their_initiations += 1
            except:
                continue
        
        # Determine who initiates more
        if my_initiations > their_initiations:
            initiation_leader = "You initiate more conversations"
        elif their_initiations > my_initiations:
            initiation_leader = "They initiate more conversations"
        else:
            initiation_leader = "Equal conversation initiation"
        
        # Time period analysis (simplified)
        most_active_period = "Throughout the day"  # Could be enhanced with hour analysis
        
        # Frequency description
        if avg_messages_per_day > 50:
            conversation_frequency = "Multiple times daily"
        elif avg_messages_per_day > 10:
            conversation_frequency = "Daily"
        elif avg_messages_per_day > 3:
            conversation_frequency = "Several times per week"
        else:
            conversation_frequency = "Weekly or less"
        
        return {
            'total_messages': total_messages,
            'my_messages': my_messages,
            'their_messages': their_messages,
            'my_percentage': my_percentage,
            'their_percentage': their_percentage,
            'avg_my_length': avg_my_length,
            'avg_their_length': avg_their_length,
            'conversation_span': conversation_span,
            'avg_messages_per_day': avg_messages_per_day,
            'my_initiations': my_initiations,
            'their_initiations': their_initiations,
            'initiation_leader': initiation_leader,
            'most_active_period': most_active_period,
            'conversation_frequency': conversation_frequency
        }
    
    def _get_comprehensive_context(self, question: str) -> List[SimpleChunk]:
        """Get comprehensive context for complex multi-step reasoning questions"""
        # Combine different types of context for comprehensive analysis
        all_context = []
        
        # Get timeline context (relationship progression)
        timeline_chunks = self._get_timeline_context(question)
        all_context.extend(timeline_chunks[:8])  # Limit each type
        
        # Get emotional context (sentiment patterns)
        emotional_chunks = self._get_emotional_context(question)
        all_context.extend(emotional_chunks[:8])
        
        # Get pattern context (behavioral data)
        pattern_chunks = self._get_pattern_context(question)
        all_context.extend(pattern_chunks[:8])
        
        # Get relationship context (general dynamics)
        relationship_chunks = self._get_relationship_context(question)
        all_context.extend(relationship_chunks[:6])
        
        # Remove duplicates while preserving different types of analysis
        seen_ids = set()
        comprehensive_context = []
        
        for chunk in all_context:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                comprehensive_context.append(chunk)
        
        # Sort by timestamp for chronological understanding
        comprehensive_context.sort(key=lambda x: x.timestamp)
        
        # Return comprehensive context (up to 25 chunks for thorough analysis)
        return comprehensive_context[:25]
    
    def search_semantic(self, query: str, top_k: int = 3) -> List[SimpleChunk]:
        """Semantic search on subset of messages"""
        # Get text search results first
        text_results = self.search_text(query, top_k * 3)
        
        if not text_results:
            return []
        
        try:
            # Encode query and chunk texts
            query_embedding = self.encoder.encode([query])
            chunk_texts = [chunk.text for chunk in text_results]
            chunk_embeddings = self.encoder.encode(chunk_texts, batch_size=8, show_progress_bar=False)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [text_results[i] for i in top_indices]
            
        except Exception as e:
            console.print(f"⚠️  Semantic search failed, using text search: {e}")
            return text_results[:top_k]
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the conversation"""
        # Detect if this is a relationship/contextual question requiring inference
        relationship_keywords = [
            'dating', 'relationship', 'together', 'couple', 'romantic', 
            'boyfriend', 'girlfriend', 'love', 'feelings', 'attracted',
            'like each other', 'going out', 'seeing each other', 'like me',
            'interested', 'flirting', 'chemistry', 'connection', 'vibe'
        ]
        
        contextual_keywords = [
            'close', 'friends', 'best friend', 'family', 'siblings',
            'meet', 'met', 'first time', 'how long', 'when did',
            'dynamic', 'relationship', 'think of', 'feel about',
            'our relationship', 'between us', 'what are we', 'status'
        ]
        
        temporal_keywords = [
            'when', 'start', 'began', 'first', 'initially', 'over time',
            'changed', 'progression', 'evolution', 'timeline', 'chronologically',
            'months ago', 'weeks ago', 'recently', 'lately', 'earlier',
            'before', 'after', 'since', 'until', 'during', 'throughout'
        ]
        
        emotional_keywords = [
            'feel', 'feeling', 'feelings', 'emotion', 'emotional', 'mood',
            'happy', 'sad', 'angry', 'excited', 'nervous', 'anxious', 'worried',
            'love', 'hate', 'like', 'dislike', 'enjoy', 'appreciate', 'care',
            'upset', 'hurt', 'disappointed', 'frustrated', 'annoyed', 'mad',
            'joy', 'happiness', 'sadness', 'anger', 'fear', 'surprise',
            'affection', 'attraction', 'chemistry', 'vibe', 'energy',
            'tone', 'attitude', 'sentiment', 'react', 'reaction'
        ]
        
        pattern_keywords = [
            'often', 'frequency', 'how much', 'how many', 'usually', 'typically',
            'initiate', 'start', 'first', 'reach out', 'contact', 'message first',
            'respond', 'response', 'reply', 'answer', 'get back', 'respond to',
            'fast', 'quick', 'slow', 'delay', 'time', 'timing', 'speed',
            'pattern', 'habit', 'routine', 'behavior', 'tendency', 'style',
            'morning', 'evening', 'night', 'late', 'early', 'weekend', 'weekday'
        ]
        
        complex_keywords = [
            'analyze', 'analysis', 'comprehensive', 'detailed', 'explain why',
            'walk me through', 'break down', 'step by step', 'evidence',
            'prove', 'support', 'because', 'reasons', 'factors', 'overall',
            'summary', 'conclusion', 'assessment', 'evaluation', 'deep dive'
        ]
        
        is_relationship_question = any(keyword in question.lower() for keyword in relationship_keywords)
        is_contextual_question = any(keyword in question.lower() for keyword in contextual_keywords)
        is_temporal_question = any(keyword in question.lower() for keyword in temporal_keywords)
        is_emotional_question = any(keyword in question.lower() for keyword in emotional_keywords)
        is_pattern_question = any(keyword in question.lower() for keyword in pattern_keywords)
        is_complex_question = any(keyword in question.lower() for keyword in complex_keywords)
        
        # Check if question needs multi-step reasoning (combines multiple aspects)
        question_aspects = [
            is_temporal_question, is_emotional_question, is_pattern_question,
            is_relationship_question, is_contextual_question
        ]
        aspect_count = sum(question_aspects)
        
        if is_complex_question or aspect_count > 1:
            # For complex questions requiring multi-step reasoning
            results = self._get_comprehensive_context(question)
        elif is_temporal_question:
            # For timeline questions, get chronological context
            results = self._get_timeline_context(question)
        elif is_emotional_question:
            # For emotional questions, get sentiment-rich context
            results = self._get_emotional_context(question)
        elif is_pattern_question:
            # For pattern questions, get behavioral context with stats
            results = self._get_pattern_context(question)
        elif is_relationship_question or is_contextual_question:
            # For relationship questions, get broader context
            results = self._get_relationship_context(question)
        else:
            # Regular semantic search
            results = self.search_semantic(question, top_k=3)
            
            if not results:
                results = self.search_text(question, top_k=3)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in your conversations.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Build context from results
        context_parts = []
        sources = []
        
        for chunk in results:
            context_parts.append(f"[{chunk.date}] {chunk.text}")
            sources.append({
                'date': chunk.date,
                'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'relevance_score': 0.8
            })
        
        context = "\n\n".join(context_parts)
        
        # Create enhanced prompt for deeper understanding
        if is_complex_question or aspect_count > 1:
            prompt = f"""You are conducting a comprehensive analysis of a relationship by examining multiple dimensions: timeline progression, emotional patterns, communication behaviors, and relationship dynamics. Use the extensive data below to provide thorough, multi-step reasoning.

Comprehensive Analysis Context:
{context}

Question: {question}

Instructions for Multi-Step Analysis:
1. TIMELINE ANALYSIS: How has the relationship evolved over time?
2. EMOTIONAL ANALYSIS: What are the sentiment patterns and emotional dynamics?  
3. BEHAVIORAL ANALYSIS: What do communication patterns reveal?
4. RELATIONSHIP ANALYSIS: What is the overall relationship dynamic?
5. SYNTHESIS: Combine all evidence to reach a well-supported conclusion

For each step:
- Use specific evidence from the conversation data
- Cite dates, statistics, and examples when available
- Show how different pieces of evidence support or contradict each other
- Build a logical chain of reasoning
- Address potential counterarguments

Provide a structured analysis that walks through your reasoning step-by-step, then give a clear final conclusion with confidence level based on the strength of evidence."""
        elif is_temporal_question:
            prompt = f"""You are analyzing a conversation timeline to understand how a relationship evolved over time. The conversation excerpts below are arranged chronologically from earliest to most recent.

Chronological Conversation Context:
{context}

Question: {question}

Instructions:
- Analyze how the conversation style, tone, and intimacy changed over time
- Look for progression markers: formal → casual, distant → close, rare → frequent contact
- Identify key moments or turning points in the relationship
- Notice changes in emoji use, language style, topic depth, response patterns
- Track the evolution of how they address each other, share personal info, make plans
- Consider seasonal patterns, life events, or relationship milestones
- Provide specific timeframes when describing changes (e.g., "starting in March 2024")
- If you can identify clear relationship progression, describe it confidently

Answer based on your chronological analysis of the relationship timeline:"""
        elif is_emotional_question:
            prompt = f"""You are analyzing conversations to understand emotional patterns, feelings, and sentiment between two people. The conversation excerpts below contain emotionally significant moments and expressions.

Emotional Conversation Context:
{context}

Question: {question}

Instructions:
- Analyze emotional indicators: word choice, emoji use, exclamation points, capitalization
- Look for sentiment patterns: positive, negative, neutral emotional expressions
- Identify emotional intensity: subtle hints vs. strong expressions of feeling  
- Notice emotional triggers: what causes joy, frustration, excitement, sadness
- Track emotional reciprocity: how emotions are shared and responded to
- Consider emotional subtext: what feelings might be implied but not directly stated
- Analyze emotional progression: how feelings have developed or changed
- Look for emotional comfort levels: vulnerability, openness, emotional sharing

Answer based on your emotional and sentiment analysis:"""
        elif is_pattern_question:
            prompt = f"""You are analyzing communication patterns and behavioral data from conversations between two people. The data below includes statistical analysis and conversation samples showing behavioral patterns.

Communication Pattern Context:
{context}

Question: {question}

Instructions:
- Use the statistical data provided to give specific, quantitative insights
- Analyze communication frequency, message initiation, response patterns
- Look for behavioral tendencies: who reaches out more, message length patterns, timing habits
- Consider communication balance: equal participation vs. one-sided communication
- Identify communication styles: formal vs. casual, lengthy vs. brief messages
- Notice timing patterns: when conversations typically happen, response speeds
- Evaluate engagement levels: who puts more effort into conversations
- Provide specific numbers and percentages when available in the data

Answer based on the communication patterns and behavioral analysis:"""
        elif is_relationship_question or is_contextual_question:
            prompt = f"""You are analyzing a conversation between two people to understand their relationship dynamics. Based on the conversation excerpts below, answer the user's question by reading between the lines and inferring meaning from context, tone, frequency, and patterns.

Conversation Context:
{context}

Question: {question}

Instructions:
- Analyze conversation patterns, tone, intimacy level, frequency of contact
- Look for subtle indicators like emoji use, response time implications, shared activities
- Consider the evolution of the relationship over time
- Read between the lines - what is implied but not explicitly stated?
- If you can infer the answer from context clues, do so confidently
- If there's truly insufficient information, say so

Answer based on your analysis of the relationship dynamics and conversation patterns:"""
        else:
            prompt = f"""Based on these conversation excerpts, answer the user's question accurately.

Conversation Context:
{context}

Question: {question}

Answer the question based on the information provided in the conversation excerpts above. Be specific and cite relevant parts of the conversation."""
        
        # Use Ollama to generate answer - prefer base model, fallback to fine-tuned
        def check_model_exists(model_name):
            try:
                import requests
                r = requests.get("http://localhost:11434/api/tags", timeout=2)
                if r.status_code == 200:
                    names = [m["name"] for m in r.json().get("models", [])]
                    return any(n.split(":")[0] == model_name.split(":")[0] for n in names)
            except Exception:
                pass
            return False
        
        # Choose model - base model for reasoning, fine-tuned as fallback
        base_model = "llama3.2:3b"
        fine_tuned_model = f"texttwin-{self.normalized}"
        
        if check_model_exists(base_model):
            model_to_use = base_model
        elif check_model_exists(fine_tuned_model):
            model_to_use = fine_tuned_model
            console.print(f"🤖 Using fine-tuned model for analysis: {model_to_use}")
        else:
            return {
                'answer': "Error: No suitable model available. Please ensure llama3.2:3b or your fine-tuned model is installed.",
                'confidence': 0.0,
                'sources': sources
            }
        
        try:
            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    'answer': answer,
                    'confidence': 0.8,  # Fixed confidence for now
                    'sources': sources,
                    'context_used': len(results)
                }
            else:
                return {
                    'answer': f"Error generating answer: {response.status_code}",
                    'confidence': 0.0,
                    'sources': sources
                }
                
        except Exception as e:
            return {
                'answer': f"Error: {e}",
                'confidence': 0.0,
                'sources': sources
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation"""
        if not self.messages:
            return {}
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversation_chunks")
        total_chunks = cursor.fetchone()[0]
        conn.close()
        
        # Basic stats
        total_messages = len(self.messages)
        my_messages = sum(1 for m in self.messages if m['is_from_me'])
        their_messages = total_messages - my_messages
        
        # Date range
        first_date = self.messages[0]['date']
        last_date = self.messages[-1]['date']
        
        return {
            'total_messages': total_messages,
            'my_messages': my_messages,
            'their_messages': their_messages,
            'first_message': first_date,
            'last_message': last_date,
            'total_chunks': total_chunks,
            'phone_number': self.phone_number
        }