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
            # Clean query for FTS - remove special characters that cause issues
            clean_query = ''.join(c for c in query if c.isalnum() or c.isspace())
            clean_query = ' '.join(clean_query.split())  # Remove extra spaces
            
            if not clean_query:
                raise Exception("Empty query after cleaning")
            
            # FTS search with individual words (avoid phrase matching issues)
            words = clean_query.split()[:5]  # Limit to 5 words
            if len(words) == 1:
                fts_query = words[0]
            else:
                fts_query = ' OR '.join(words)
            
            cursor.execute("""
                SELECT c.chunk_id, c.text, c.date, c.timestamp
                FROM chunks_fts 
                JOIN conversation_chunks c ON chunks_fts.chunk_id = c.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """, (fts_query, top_k))
            
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
        # Get diverse conversation samples for context-rich analysis
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get recent conversations (most relevant context)
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            ORDER BY timestamp DESC
            LIMIT 15
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
        
        # Get representative samples from different time periods
        total_chunks = cursor.execute("SELECT COUNT(*) FROM conversation_chunks").fetchone()[0]
        
        # Early period chunks
        early_chunks = []
        if total_chunks > 30:
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp
                FROM conversation_chunks
                ORDER BY timestamp ASC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                chunk = SimpleChunk(
                    chunk_id=row[0],
                    text=row[1],
                    date=row[2],
                    timestamp=row[3]
                )
                early_chunks.append(chunk)
        
        # Middle period chunks
        middle_chunks = []
        if total_chunks > 50:
            offset = total_chunks // 2
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp
                FROM conversation_chunks
                ORDER BY timestamp ASC
                LIMIT 10 OFFSET ?
            """, (offset,))
            
            for row in cursor.fetchall():
                chunk = SimpleChunk(
                    chunk_id=row[0],
                    text=row[1],
                    date=row[2],
                    timestamp=row[3]
                )
                middle_chunks.append(chunk)
        
        conn.close()
        
        # Combine all chunks for comprehensive context
        all_chunks = recent_chunks + early_chunks + middle_chunks
        seen_ids = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        # Sort by timestamp for chronological understanding
        unique_chunks.sort(key=lambda x: x.timestamp)
        
        # Return representative sample for context
        return unique_chunks[:15]
    
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
    
    def _analyze_relationship_progression(self) -> Dict[str, Any]:
        """Analyze relationship progression using LLM inference"""
        if not self.messages:
            return {}
        
        # Sample representative conversations across different time periods
        total_messages = len(self.messages)
        
        # Get early, middle, and recent conversations for progression analysis
        sample_chunks = []
        
        # Early period (first 20% of messages)
        early_end = max(1, total_messages // 5)
        early_chunk = self._create_sample_chunk(self.messages[:early_end], "early")
        if early_chunk:
            sample_chunks.append(early_chunk)
        
        # Middle period (around 40-60% of messages)
        if total_messages > 10:
            mid_start = total_messages * 2 // 5
            mid_end = total_messages * 3 // 5
            mid_chunk = self._create_sample_chunk(self.messages[mid_start:mid_end], "middle")
            if mid_chunk:
                sample_chunks.append(mid_chunk)
        
        # Recent period (last 20% of messages)
        recent_start = total_messages * 4 // 5
        recent_chunk = self._create_sample_chunk(self.messages[recent_start:], "recent")
        if recent_chunk:
            sample_chunks.append(recent_chunk)
        
        # Create analysis prompt
        conversation_samples = "\n\n".join([
            f"=== {chunk['period'].upper()} CONVERSATIONS ===\n{chunk['text']}"
            for chunk in sample_chunks
        ])
        
        prompt = f"""Analyze this conversation history to understand the relationship progression. Do not use predefined categories or keywords - infer naturally from the conversation content, tone, and evolution.

{conversation_samples}

Analyze the relationship progression and provide insights in JSON format:
{{
    "current_status": "brief description of current relationship status",
    "status_confidence": 0.0-1.0,
    "progression_summary": "how the relationship has evolved over time",
    "communication_evolution": "how their communication style/tone has changed",
    "key_indicators": ["list of specific conversational patterns that indicate relationship nature"],
    "milestones": [
        {{"date": "YYYY-MM-DD", "description": "significant relationship milestone observed"}}
    ]
}}

Base your analysis ONLY on conversational patterns, tone changes, topics discussed, and natural communication evolution. Do not rely on specific words or predefined categories."""
        
        try:
            result = self._query_llm(prompt, temperature=0.2)
            
            # Parse JSON response
            import json
            analysis = json.loads(result)
            
            # Add metadata
            analysis['total_messages'] = total_messages
            analysis['analysis_method'] = 'llm_inference'
            
            return analysis
            
        except Exception as e:
            # Fallback to basic analysis if LLM fails
            return {
                'current_status': 'Unable to analyze - insufficient data',
                'status_confidence': 0.0,
                'progression_summary': f'Analysis failed: {str(e)}',
                'communication_evolution': 'Unable to determine',
                'key_indicators': [],
                'milestones': [],
                'total_messages': total_messages,
                'analysis_method': 'fallback'
            }
    
    def _create_sample_chunk(self, messages: List[Dict], period: str) -> Dict[str, str]:
        """Create a representative sample chunk from a message period"""
        if not messages:
            return None
        
        # Take a sample of messages to avoid overwhelming the LLM
        sample_size = min(15, len(messages))
        step = max(1, len(messages) // sample_size)
        sampled_messages = messages[::step][:sample_size]
        
        # Format conversations
        chunk_parts = []
        for msg in sampled_messages:
            sender = "You" if msg['is_from_me'] else "Them"
            chunk_parts.append(f"[{msg['date']}] {sender}: {msg['text']}")
        
        return {
            'period': period,
            'text': '\n'.join(chunk_parts)
        }
    
    def _generate_recommendations(self, progression: Dict, patterns: Dict) -> List[Dict]:
        """Generate actionable relationship recommendations using LLM analysis"""
        if not progression or not patterns:
            return []
        
        # Create context for recommendation generation
        context = f"""
Relationship Analysis:
- Current Status: {progression.get('current_status', 'Unknown')}
- Confidence: {progression.get('status_confidence', 0):.1%}
- Communication Evolution: {progression.get('communication_evolution', 'Unknown')}
- Key Indicators: {', '.join(progression.get('key_indicators', []))}

Communication Patterns:
- Total Messages: {patterns.get('total_messages', 0)}
- Your Messages: {patterns.get('my_messages', 0)} ({patterns.get('my_percentage', 0):.1%})
- Their Messages: {patterns.get('their_messages', 0)} ({patterns.get('their_percentage', 0):.1%})
- Average Response Time: {patterns.get('avg_response_time_hours', 0):.1f} hours
- Communication Frequency: {patterns.get('messages_per_day', 0):.1f} messages/day
"""
        
        prompt = f"""Based on this relationship and communication analysis, provide 3-5 actionable recommendations for improving the relationship.

{context}

Generate recommendations in JSON format:
[
    {{
        "title": "specific actionable title",
        "description": "detailed recommendation with specific actions",
        "priority": "high|medium|low",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation of why this recommendation matters"
    }}
]

Focus on practical, specific advice that considers the natural evolution of the relationship. Avoid generic advice."""
        
        try:
            result = self._query_llm(prompt, temperature=0.3)
            
            # Parse JSON response
            import json
            recommendations = json.loads(result)
            
            return recommendations
            
        except Exception as e:
            # Fallback recommendations if LLM fails
            return [{
                'title': 'Analysis Error',
                'description': f'Unable to generate personalized recommendations: {str(e)}',
                'priority': 'low',
                'confidence': 0.0,
                'reasoning': 'LLM analysis failed'
            }]
    
    def _generate_relationship_summary(self, progression: Dict, patterns: Dict) -> str:
        """Generate a comprehensive relationship summary using LLM analysis"""
        if not progression or not patterns:
            return "Insufficient data for analysis"
        
        # Create context for summary generation
        context = f"""
Relationship Analysis Results:
- Current Status: {progression.get('current_status', 'Unknown')}
- Confidence: {progression.get('status_confidence', 0):.1%}
- Progression Summary: {progression.get('progression_summary', 'N/A')}
- Communication Evolution: {progression.get('communication_evolution', 'N/A')}
- Key Indicators: {', '.join(progression.get('key_indicators', []))}

Communication Statistics:
- Total Messages: {patterns.get('total_messages', 0):,}
- Your Messages: {patterns.get('my_messages', 0)} ({patterns.get('my_percentage', 0):.1%})
- Their Messages: {patterns.get('their_messages', 0)} ({patterns.get('their_percentage', 0):.1%})
- Conversation Span: {patterns.get('conversation_span', 0)} days
- Average Response Time: {patterns.get('avg_response_time_hours', 0):.1f} hours
- Daily Message Frequency: {patterns.get('messages_per_day', 0):.1f}

Milestones Identified: {len(progression.get('milestones', []))} key moments
"""
        
        prompt = f"""Create a comprehensive yet concise relationship summary based on this analysis. 

{context}

Write a natural, insightful summary (2-3 paragraphs) that:
1. Describes the current relationship status and how it evolved
2. Highlights key communication patterns and what they reveal
3. Provides context about the relationship's trajectory

Be specific and analytical, not generic. Focus on what makes this relationship unique based on the conversation patterns observed."""
        
        try:
            result = self._query_llm(prompt, temperature=0.4)
            return result.strip()
            
        except Exception as e:
            # Fallback to basic summary if LLM fails
            return f"""**RELATIONSHIP ANALYSIS SUMMARY**

Status: {progression.get('current_status', 'Unknown')} (Confidence: {progression.get('status_confidence', 0):.1%})
Total Messages: {patterns.get('total_messages', 0):,} over {patterns.get('conversation_span', 0)} days
Communication Balance: {patterns.get('my_percentage', 0):.1f}% you, {patterns.get('their_percentage', 0):.1f}% them

Analysis Method: {progression.get('analysis_method', 'LLM inference')}
Note: Detailed analysis unavailable due to processing error: {str(e)}"""
    
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
        """Answer a question about the conversation using intelligent context selection"""
        # Use LLM to classify the question type and determine best context strategy
        classification_prompt = f"""Analyze this question and classify what type of context would be most helpful to answer it:

Question: "{question}"

Classify as one of:
1. "timeline" - Questions about when things happened, chronological progression, time-based changes
2. "emotional" - Questions about feelings, emotions, sentiment, mood, reactions  
3. "pattern" - Questions about communication habits, behaviors, frequency, response patterns
4. "relationship" - Questions about relationship dynamics, status, connection type
5. "comprehensive" - Complex questions requiring multiple types of context
6. "factual" - Simple factual questions that can be answered with basic search

Respond with just the classification word."""
        
        try:
            question_type = self._query_llm(classification_prompt, temperature=0.1).strip().lower()
            console.print(f"🔍 Question classified as: {question_type}")
        except Exception as e:
            console.print(f"⚠️  Classification failed: {e}")
            question_type = "comprehensive"  # Default to comprehensive if classification fails
        
        # Select appropriate context retrieval strategy based on classification
        results = []
        
        try:
            if question_type == "timeline":
                results = self._get_timeline_context(question)
            elif question_type == "emotional":
                results = self._get_emotional_context(question)
            elif question_type == "pattern":
                results = self._get_pattern_context(question)
            elif question_type == "relationship":
                results = self._get_relationship_context(question)
            elif question_type == "comprehensive":
                results = self._get_comprehensive_context(question)
            else:  # factual
                # For simple factual questions, use direct search
                results = self.search_semantic(question, top_k=5)
                if not results:
                    results = self.search_text(question, top_k=5)
        except Exception as e:
            console.print(f"⚠️  Context retrieval failed: {e}")
            # Fallback to comprehensive context
            try:
                results = self._get_comprehensive_context(question)
            except:
                # Final fallback to simple search
                results = self.search_semantic(question, top_k=10)
                if not results:
                    results = self.search_text(question, top_k=10)
        
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
        
        # Create intelligent prompt based on question type
        analysis_instructions = {
            "timeline": "Focus on chronological progression, changes over time, and temporal patterns.",
            "emotional": "Analyze emotional tone, sentiment patterns, feelings, and reactions.",
            "pattern": "Examine communication behaviors, habits, frequencies, and interaction patterns.",  
            "relationship": "Assess relationship dynamics, connection type, and interpersonal context.",
            "comprehensive": "Provide thorough analysis considering multiple dimensions: timeline, emotions, patterns, and relationships.",
            "factual": "Answer directly based on the conversation content."
        }
        
        instruction = analysis_instructions.get(question_type, analysis_instructions["comprehensive"])
        
        prompt = f"""Based on these conversation excerpts, answer the user's question with accuracy and insight.

Conversation Context:
{context}

Question: {question}

Analysis Focus: {instruction}

Instructions:
- Base your answer ONLY on the provided conversation excerpts
- Use specific evidence and examples from the conversations
- If relevant information isn't available, clearly state this
- Provide a confidence level based on the strength of available evidence
- Be analytical but conversational in your response

Answer the question thoroughly and honestly based on what you can observe in the conversation data."""
        
        # Use Ollama to generate answer
        try:
            result = self._query_llm(prompt, temperature=0.3)
            
            return {
                'answer': result.strip(),
                'confidence': 0.8,  # Confidence based on context quality
                'sources': sources,
                'context_used': len(results)
            }
            
        except Exception as e:
            return {
                'answer': f"Error analyzing conversation: {str(e)}",
                'confidence': 0.0,
                'sources': sources
            }
    
    def _query_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Query the LLM with a prompt and return the response"""
        import requests
        
        # Choose model - prefer base model for analysis, fallback to fine-tuned
        base_model = "llama3.2:3b"
        fine_tuned_model = f"texttwin-{self.normalized}"
        
        def check_model_exists(model_name):
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=2)
                if r.status_code == 200:
                    names = [m["name"] for m in r.json().get("models", [])]
                    return any(n.split(":")[0] == model_name.split(":")[0] for n in names)
            except Exception:
                pass
            return False
        
        if check_model_exists(base_model):
            model_to_use = base_model
        elif check_model_exists(fine_tuned_model):
            model_to_use = fine_tuned_model
        else:
            raise Exception("No suitable model available. Please ensure llama3.2:3b is installed.")
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            raise Exception(f"LLM request failed with status {response.status_code}")
    
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