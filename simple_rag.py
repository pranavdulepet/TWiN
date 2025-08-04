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
        
        is_relationship_question = any(keyword in question.lower() for keyword in relationship_keywords)
        is_contextual_question = any(keyword in question.lower() for keyword in contextual_keywords)
        
        if is_relationship_question or is_contextual_question:
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
        if is_relationship_question or is_contextual_question:
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
        
        # Use Ollama to generate answer
        try:
            payload = {
                "model": "llama3.2:3b",
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