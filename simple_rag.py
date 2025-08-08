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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

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
        
        # Ensure data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Paths
        self.conv_file = f"data/conversation_{self.normalized}.json"
        self.db_file = f"data/simple_rag_{self.normalized}.db"
        
        # Load data
        self.messages = []
        self.chunks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            
            init_task = progress.add_task("🧠 Initializing RAG system...", total=100)
            
            # Initialize embedding model (upgraded to SOTA model)
            progress.update(init_task, advance=20, description="🤖 Loading advanced embedding model...")
            try:
                # Try best-in-class model first
                self.encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
                console.print("✨ Using BGE-small-en-v1.5 (state-of-the-art)")
            except Exception as e:
                try:
                    # Fallback to high-quality alternative
                    self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                    console.print("✨ Using all-mpnet-base-v2 (high quality)")
                except Exception as e2:
                    # Final fallback to original
                    self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                    console.print("⚠️  Using all-MiniLM-L6-v2 (fallback)")
            
            progress.update(init_task, advance=30, description="📁 Loading conversation data...")
            self._load_conversation()
            
            progress.update(init_task, advance=20, description="🗄️ Setting up database...")
            self._setup_database()
            
            progress.update(init_task, advance=30, description="🔨 Creating conversation chunks...")
            self._create_chunks()
            
            progress.update(init_task, completed=100, description="✅ RAG system ready")
    
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
        """Create conversation-aware chunks with intelligent boundaries"""
        # Check if chunks already exist
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversation_chunks")
        existing_count = cursor.fetchone()[0]
        conn.close()
        
        if existing_count > 0:
            console.print(f"📚 Using existing {existing_count} chunks")
            return
        
        if not self.messages:
            console.print("⚠️  No messages to create chunks from")
            return
        
        # Create conversation-aware chunks
        chunks = self._smart_chunking()
        
        if not chunks:
            console.print("⚠️  No chunks created")
            return
        
        # Get total for progress tracking
        total_chunks = len(chunks)
        
        # Simple progress without Rich Progress (to avoid conflicts)
        console.print(f"🔨 Creating {total_chunks} conversation chunks...")
        
        # Store in database
        console.print(f"💾 Storing {len(chunks)} chunks in database...")
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        for i, chunk in enumerate(chunks):
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
        
        console.print(f"✅ Created and indexed {len(chunks)} conversation chunks")
    
    def _smart_chunking(self) -> List[SimpleChunk]:
        """Create conversation-aware chunks with intelligent boundaries"""
        from datetime import datetime, timedelta
        
        if not self.messages:
            return []
        
        chunks = []
        current_chunk_messages = []
        
        # Parameters for smart chunking
        max_chunk_size = 15  # Maximum messages per chunk
        min_chunk_size = 3   # Minimum messages per chunk
        time_gap_hours = 4   # Hours gap to consider new conversation
        
        for i, message in enumerate(self.messages):
            current_chunk_messages.append(message)
            
            # Check if we should end this chunk
            should_end_chunk = False
            
            # 1. Reached maximum chunk size
            if len(current_chunk_messages) >= max_chunk_size:
                should_end_chunk = True
            
            # 2. Large time gap indicates conversation break
            elif len(current_chunk_messages) >= min_chunk_size and i < len(self.messages) - 1:
                try:
                    current_time = datetime.strptime(message['date'], '%Y-%m-%d %H:%M:%S')
                    next_time = datetime.strptime(self.messages[i + 1]['date'], '%Y-%m-%d %H:%M:%S')
                    time_diff = (next_time - current_time).total_seconds() / 3600
                    
                    if time_diff > time_gap_hours:
                        should_end_chunk = True
                except:
                    pass  # Skip time-based chunking if date parsing fails
            
            # 3. Topic change detection (simple heuristic)
            elif len(current_chunk_messages) >= min_chunk_size and i < len(self.messages) - 1:
                # Look for conversation restarts (greetings after gaps)
                next_message = self.messages[i + 1]['text'].lower()
                greetings = ['hi', 'hey', 'hello', 'good morning', 'good night', 'how are you']
                
                if any(greeting in next_message[:20] for greeting in greetings):
                    should_end_chunk = True
            
            # Create chunk if we should end it or if this is the last message
            if should_end_chunk or i == len(self.messages) - 1:
                if current_chunk_messages:
                    chunk = self._create_chunk_from_messages(current_chunk_messages, len(chunks))
                    chunks.append(chunk)
                    current_chunk_messages = []
        
        return chunks
    
    def _create_chunk_from_messages(self, messages: List[Dict], chunk_index: int) -> SimpleChunk:
        """Create a chunk from a list of messages"""
        if not messages:
            return None
        
        # Create chunk text
        chunk_parts = []
        for msg in messages:
            sender = "You" if msg['is_from_me'] else "Them"
            chunk_parts.append(f"{sender}: {msg['text']}")
        
        chunk_text = "\n".join(chunk_parts)
        
        # Use first and last message for chunk metadata
        start_msg = messages[0]
        end_msg = messages[-1]
        
        chunk = SimpleChunk(
            chunk_id=f"smart_chunk_{chunk_index}_{start_msg['timestamp']}_{end_msg['timestamp']}",
            text=chunk_text,
            date=start_msg['date'],
            timestamp=start_msg['timestamp']
        )
        
        return chunk
    
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
    
    def search_hybrid(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[SimpleChunk]:
        """Advanced hybrid search combining semantic, lexical, and temporal relevance"""
        # Get results from multiple search methods
        semantic_results = self._search_semantic_only(query, top_k * 2)
        text_results = self.search_text(query, top_k * 2)
        
        # Combine and rank results
        combined_results = self._fusion_ranking(
            semantic_results, text_results, query, 
            semantic_weight=semantic_weight, top_k=top_k
        )
        
        return combined_results
    
    def _search_semantic_only(self, query: str, top_k: int = 10) -> List[tuple]:
        """Pure semantic search returning (chunk, score) tuples"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get all chunks for semantic search
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp
            FROM conversation_chunks
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        chunks = []
        for row in cursor.fetchall():
            chunk = SimpleChunk(
                chunk_id=row[0], text=row[1], date=row[2], timestamp=row[3]
            )
            chunks.append(chunk)
        
        conn.close()
        
        if not chunks:
            return []
        
        try:
            # Encode query and all chunk texts
            query_embedding = self.encoder.encode([query])
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_embeddings = self.encoder.encode(chunk_texts, batch_size=8, show_progress_bar=False)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
            
            # Return top results with scores
            scored_results = [(chunks[i], float(similarities[i])) for i in range(len(chunks))]
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            return scored_results[:top_k]
            
        except Exception as e:
            console.print(f"⚠️  Semantic search failed: {e}")
            return [(chunk, 0.5) for chunk in chunks[:top_k]]
    
    def _fusion_ranking(self, semantic_results: List[tuple], text_results: List[SimpleChunk], 
                       query: str, semantic_weight: float = 0.7, top_k: int = 5) -> List[SimpleChunk]:
        """Intelligent fusion of different search methods"""
        
        # Create scoring dictionary
        chunk_scores = {}
        
        # Add semantic scores (normalized)
        if semantic_results:
            max_semantic_score = max(score for _, score in semantic_results) if semantic_results else 1.0
            for chunk, score in semantic_results:
                normalized_score = score / max_semantic_score if max_semantic_score > 0 else 0
                chunk_scores[chunk.chunk_id] = {
                    'chunk': chunk,
                    'semantic_score': normalized_score * semantic_weight,
                    'text_score': 0.0,
                    'temporal_bonus': 0.0
                }
        
        # Add text search scores
        text_weight = 1.0 - semantic_weight
        for i, chunk in enumerate(text_results):
            # Text search score based on rank (higher rank = higher score)
            text_score = (len(text_results) - i) / len(text_results) * text_weight
            
            if chunk.chunk_id in chunk_scores:
                chunk_scores[chunk.chunk_id]['text_score'] = text_score
            else:
                chunk_scores[chunk.chunk_id] = {
                    'chunk': chunk,
                    'semantic_score': 0.0,
                    'text_score': text_score,
                    'temporal_bonus': 0.0
                }
        
        # Add temporal bonus (recent conversations get slight boost)
        from datetime import datetime, timedelta
        current_time = datetime.now().timestamp()
        
        for chunk_id, data in chunk_scores.items():
            chunk = data['chunk']
            try:
                chunk_time = datetime.strptime(chunk.date.split()[0], '%Y-%m-%d').timestamp()
                days_ago = (current_time - chunk_time) / (24 * 3600)
                
                # Slight bonus for recent conversations (decays over 30 days)
                temporal_bonus = max(0, (30 - days_ago) / 30) * 0.1
                data['temporal_bonus'] = temporal_bonus
            except:
                data['temporal_bonus'] = 0.0
        
        # Calculate final scores and rank
        final_results = []
        for chunk_id, data in chunk_scores.items():
            total_score = (data['semantic_score'] + data['text_score'] + data['temporal_bonus'])
            final_results.append((data['chunk'], total_score))
        
        # Sort by total score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in final_results[:top_k]]
    
    def search_semantic(self, query: str, top_k: int = 3) -> List[SimpleChunk]:
        """Legacy method - now uses hybrid search"""
        return self.search_hybrid(query, top_k)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the conversation using intelligent context selection"""
        # Enhanced query classification with local analysis
        question_type = self._classify_query_locally(question)
        console.print(f"🔍 Query classified as: {question_type}")
        
        # Use appropriate search strategy based on question type
        if question_type in ['timeline', 'temporal']:
            results = self._get_timeline_context(question)
        elif question_type in ['emotional', 'sentiment']:
            results = self._get_emotional_context(question)
        elif question_type in ['pattern', 'behavioral']:
            results = self._get_pattern_context(question)
        elif question_type in ['relationship', 'social']:
            results = self._get_relationship_context(question)
        elif question_type in ['factual', 'specific']:
            # Use hybrid search for factual questions
            results = self.search_hybrid(question, top_k=7)
        else:  # comprehensive or unknown
            results = self._get_comprehensive_context(question)
        
        # Handle the case where results is a list vs other types
        if isinstance(results, list) and not results:
            return {
                'answer': "I couldn't find any relevant information in your conversations.",
                'confidence': 0.0,
                'sources': []
            }
        elif not results:
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
    
    def _classify_query_locally(self, question: str) -> str:
        """Fast local query classification using keywords and patterns"""
        question_lower = question.lower()
        
        # Timeline/temporal keywords
        timeline_keywords = ['when', 'date', 'time', 'ago', 'last', 'first', 'recent', 'before', 'after', 
                           'yesterday', 'today', 'week', 'month', 'year', 'timeline', 'chronological']
        
        # Emotional/sentiment keywords
        emotional_keywords = ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited', 'love', 'hate',
                            'mood', 'sentiment', 'reaction', 'upset', 'joy', 'worried', 'anxious']
        
        # Pattern/behavioral keywords  
        pattern_keywords = ['often', 'usually', 'always', 'never', 'frequency', 'habit', 'pattern',
                          'tend to', 'typically', 'generally', 'behavior', 'style', 'way']
        
        # Relationship keywords
        relationship_keywords = ['relationship', 'connection', 'close', 'friend', 'dating', 'couple',
                               'dynamic', 'between us', 'together', 'apart', 'chemistry']
        
        # Factual keywords
        factual_keywords = ['what', 'who', 'where', 'which', 'specific', 'exactly', 'tell me about']
        
        # Count keyword matches
        timeline_score = sum(1 for keyword in timeline_keywords if keyword in question_lower)
        emotional_score = sum(1 for keyword in emotional_keywords if keyword in question_lower)
        pattern_score = sum(1 for keyword in pattern_keywords if keyword in question_lower)
        relationship_score = sum(1 for keyword in relationship_keywords if keyword in question_lower)
        factual_score = sum(1 for keyword in factual_keywords if keyword in question_lower)
        
        # Determine classification based on highest score
        scores = {
            'timeline': timeline_score,
            'emotional': emotional_score,
            'pattern': pattern_score,
            'relationship': relationship_score,
            'factual': factual_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'comprehensive'  # No specific keywords found
        
        # Return the type with highest score
        for qtype, score in scores.items():
            if score == max_score:
                return qtype
        
        return 'comprehensive'  # Fallback
        
        # Handle the case where results is a list vs other types
        if isinstance(results, list) and not results:
            return {
                'answer': "I couldn't find any relevant information in your conversations.",
                'confidence': 0.0,
                'sources': []
            }
        elif not results:
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
        preferred_models = ["gpt-oss:20b", "llama3.2:3b"]
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
        
        def test_model_health(model_name):
            try:
                payload = {"model": model_name, "prompt": "Hi", "stream": False, "options": {"max_tokens": 5}}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
                return response.status_code == 200 and "error" not in response.text.lower()
            except Exception:
                return False
        
        # Try preferred models in order
        model_to_use = None
        for model in preferred_models:
            if check_model_exists(model) and test_model_health(model):
                model_to_use = model
                break
        
        # Fallback to fine-tuned model if available
        if not model_to_use and check_model_exists(fine_tuned_model) and test_model_health(fine_tuned_model):
            model_to_use = fine_tuned_model
        
        # Final fallback
        if not model_to_use:
            model_to_use = "llama3.2:3b"  # Assume this works as last resort
        
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