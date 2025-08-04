#!/usr/bin/env python3
"""
RAG System for Conversation Memory and Q&A
Provides incredibly accurate retrieval of conversation history
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@dataclass
class ConversationChunk:
    """Represents a searchable conversation chunk"""
    chunk_id: str
    text: str
    date: str
    timestamp: int
    is_from_me: bool
    context_before: List[Dict]
    context_after: List[Dict]
    embedding: Optional[np.ndarray] = None

class ConversationRAG:
    """RAG system for conversation memory and Q&A"""
    
    def __init__(self, phone_number: str, chunk_size: int = 5, overlap: int = 1):
        self.phone_number = phone_number
        self.normalized = phone_number.replace('(', '').replace(')', '').replace(' ', '').replace('-', '')
        self.chunk_size = chunk_size  # Messages per chunk (increased to reduce total chunks)
        self.overlap = overlap  # Overlapping messages between chunks
        
        # Initialize embedding model
        console.print("🧠 Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Paths
        self.conv_file = f"data/conversation_{self.normalized}.json"
        self.db_file = f"data/rag_{self.normalized}.db"
        self.index_file = f"data/rag_{self.normalized}.faiss"
        
        # Load data
        self.messages = []
        self.chunks = []
        self.index = None
        
        self._load_conversation()
        self._setup_database()
        self._load_or_create_index()
    
    def _load_conversation(self):
        """Load conversation messages"""
        if not Path(self.conv_file).exists():
            console.print(f"❌ Conversation file not found: {self.conv_file}")
            return
            
        with open(self.conv_file, 'r') as f:
            self.messages = json.load(f)
        
        console.print(f"📱 Loaded {len(self.messages)} messages")
    
    def _setup_database(self):
        """Setup SQLite database for metadata storage"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                is_from_me BOOLEAN NOT NULL,
                context_before TEXT,
                context_after TEXT,
                message_ids TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_chunks(timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date ON conversation_chunks(date);
        """)
        
        conn.commit()
        conn.close()
    
    def _create_chunks(self) -> List[ConversationChunk]:
        """Create conversation chunks for embedding"""
        chunks = []
        
        for i in range(0, len(self.messages), self.chunk_size - self.overlap):
            end_idx = min(i + self.chunk_size, len(self.messages))
            chunk_messages = self.messages[i:end_idx]
            
            if not chunk_messages:
                continue
            
            # Create chunk text
            chunk_parts = []
            for msg in chunk_messages:
                sender = "You" if msg['is_from_me'] else "Them"
                chunk_parts.append(f"{sender}: {msg['text']}")
            
            chunk_text = "\n".join(chunk_parts)
            
            # Get context
            context_before = self.messages[max(0, i-5):i] if i > 0 else []
            context_after = self.messages[end_idx:end_idx+5] if end_idx < len(self.messages) else []
            
            # Create chunk
            chunk = ConversationChunk(
                chunk_id=f"chunk_{i}_{end_idx}",
                text=chunk_text,
                date=chunk_messages[0]['date'],
                timestamp=chunk_messages[0]['timestamp'],
                is_from_me=chunk_messages[0]['is_from_me'],
                context_before=context_before,
                context_after=context_after
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _store_chunks_in_db(self, chunks: List[ConversationChunk]):
        """Store chunks in database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Clear existing chunks
        cursor.execute("DELETE FROM conversation_chunks")
        
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO conversation_chunks 
                (chunk_id, text, date, timestamp, is_from_me, context_before, context_after, message_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.text,
                chunk.date,
                chunk.timestamp,
                chunk.is_from_me,
                json.dumps(chunk.context_before),
                json.dumps(chunk.context_after),
                ""  # We can add message IDs later if needed
            ))
        
        conn.commit()
        conn.close()
    
    def _create_embeddings(self, chunks: List[ConversationChunk]) -> np.ndarray:
        """Create embeddings for all chunks"""
        console.print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        
        try:
            texts = [chunk.text for chunk in chunks]
            
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            embeddings = self.encoder.encode(
                texts, 
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            
            # Store embeddings in chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
            
            return embeddings
            
        except Exception as e:
            console.print(f"❌ Error creating embeddings: {e}")
            raise
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index for similarity search"""
        try:
            console.print("🔨 Creating FAISS index...")
            dimension = embeddings.shape[1]
            
            # Use a more memory-efficient index for large datasets
            if len(embeddings) > 50000:
                # Use IVF index for large datasets
                nlist = min(4096, len(embeddings) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                embeddings_f32 = embeddings.astype('float32')
                
                # Train the index
                console.print("🎯 Training index...")
                index.train(embeddings_f32)
                index.add(embeddings_f32)
                index.nprobe = min(128, nlist // 4)  # Search parameters
            else:
                # Use flat index for smaller datasets
                index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(embeddings)
                index.add(embeddings.astype('float32'))
            
            console.print("✅ FAISS index created successfully")
            return index
            
        except Exception as e:
            console.print(f"❌ Error creating FAISS index: {e}")
            raise
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if Path(self.db_file).exists() and Path(self.index_file).exists():
            try:
                # Load from existing files
                self.index = faiss.read_index(self.index_file)
                self.chunks = self._load_chunks_from_db()
                console.print(f"📚 Loaded existing index with {len(self.chunks)} chunks")
                return
            except Exception as e:
                console.print(f"⚠️  Failed to load existing index: {e}")
        
        # Create new index
        console.print("🔨 Creating new conversation index...")
        self.chunks = self._create_chunks()
        
        if not self.chunks:
            console.print("❌ No chunks created")
            return
        
        # Store in database
        self._store_chunks_in_db(self.chunks)
        
        # Create embeddings and index
        embeddings = self._create_embeddings(self.chunks)
        self.index = self._create_faiss_index(embeddings)
        
        # Save index
        faiss.write_index(self.index, self.index_file)
        
        # Clean up memory
        import gc
        gc.collect()
        
        console.print(f"✅ Created index with {len(self.chunks)} chunks")
    
    def _load_chunks_from_db(self) -> List[ConversationChunk]:
        """Load chunks from database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_id, text, date, timestamp, is_from_me, context_before, context_after
            FROM conversation_chunks
            ORDER BY timestamp
        """)
        
        chunks = []
        for row in cursor.fetchall():
            chunk = ConversationChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3],
                is_from_me=bool(row[4]),
                context_before=json.loads(row[5]),
                context_after=json.loads(row[6])
            )
            chunks.append(chunk)
        
        conn.close()
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[ConversationChunk, float]]:
        """Search for relevant conversation chunks"""
        if not self.index or not self.chunks:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_by_date(self, start_date: str, end_date: str = None) -> List[ConversationChunk]:
        """Search chunks by date range"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        if end_date:
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp, is_from_me, context_before, context_after
                FROM conversation_chunks
                WHERE date BETWEEN ? AND ?
                ORDER BY timestamp
            """, (start_date, end_date))
        else:
            cursor.execute("""
                SELECT chunk_id, text, date, timestamp, is_from_me, context_before, context_after
                FROM conversation_chunks
                WHERE date LIKE ?
                ORDER BY timestamp
            """, (f"{start_date}%",))
        
        chunks = []
        for row in cursor.fetchall():
            chunk = ConversationChunk(
                chunk_id=row[0],
                text=row[1],
                date=row[2],
                timestamp=row[3],
                is_from_me=bool(row[4]),
                context_before=json.loads(row[5]),
                context_after=json.loads(row[6])
            )
            chunks.append(chunk)
        
        conn.close()
        return chunks
    
    def answer_question(self, question: str, max_context_chunks: int = 5) -> Dict[str, Any]:
        """Answer a question about the conversation"""
        # Search for relevant chunks
        results = self.search(question, top_k=max_context_chunks)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in your conversations.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Build context from relevant chunks
        context_parts = []
        sources = []
        
        for chunk, score in results:
            context_parts.append(f"[{chunk.date}] {chunk.text}")
            sources.append({
                'date': chunk.date,
                'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'relevance_score': score
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for answering
        prompt = f"""Based on these conversation excerpts, answer the user's question accurately.

Conversation Context:
{context}

Question: {question}

Answer the question based ONLY on the information provided in the conversation excerpts above. If the information isn't available, say so. Be specific and cite the relevant parts of the conversation."""
        
        # Use Ollama to generate answer
        try:
            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more factual responses
                    "top_p": 0.9
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    'answer': answer,
                    'confidence': results[0][1] if results else 0.0,
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
        """Get statistics about the conversation and index"""
        if not self.messages:
            return {}
        
        # Basic stats
        total_messages = len(self.messages)
        my_messages = sum(1 for m in self.messages if m['is_from_me'])
        their_messages = total_messages - my_messages
        
        # Date range
        first_date = self.messages[0]['date']
        last_date = self.messages[-1]['date']
        
        # Index stats
        total_chunks = len(self.chunks) if self.chunks else 0
        
        return {
            'total_messages': total_messages,
            'my_messages': my_messages,
            'their_messages': their_messages,
            'first_message': first_date,
            'last_message': last_date,
            'total_chunks': total_chunks,
            'phone_number': self.phone_number
        }