#!/usr/bin/env python3
"""
Clean message extraction with proper NSKeyedArchiver decoding
"""

import sqlite3
import json
import re
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()

def decode_attributed_body(blob_data):
    """Decode NSKeyedArchiver attributedBody to extract actual text"""
    if not blob_data:
        return None
    
    try:
        # Convert bytes to string for pattern matching
        blob_str = blob_data.decode('utf-8', errors='ignore')
        
        # Pattern to extract text from NSKeyedArchiver format
        # Look for text between NSString markers
        patterns = [
            r'NSString.*?\x01\x01(.+?)\x02',  # Main pattern
            r'NSString.*?\x01(.+?)\x02', 
            r'\x01\x01(.+?)\x02',
            r'NSString.*?@(.+?)\x00',
            r'@(.+?)\x00',
            r'NSString.*?\x01(.+?)[\x00-\x02]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, blob_str, re.DOTALL)
            for match in matches:
                # Clean up the extracted text
                cleaned = match.strip()
                # Remove control characters but keep normal punctuation
                cleaned = re.sub(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]', '', cleaned)
                if len(cleaned) > 0 and not cleaned.isspace():
                    return cleaned
        
        return None
        
    except Exception as e:
        return None

def extract_conversation(phone_number: str) -> list:
    """Extract all messages for a specific phone number"""
    
    chat_db = Path.home() / "Library" / "Messages" / "chat.db"
    
    console.print(f"📱 Extracting conversation with: {phone_number}")
    
    # Normalize phone number patterns
    normalized = re.sub(r'[^\d]', '', phone_number)
    patterns = [
        f"%{normalized}%",
        f"%+1{normalized}%", 
        f"%({normalized[:3]}) {normalized[3:6]}-{normalized[6:]}%",
        normalized,
        f"+1{normalized}",
        f"({normalized[:3]}) {normalized[3:6]}-{normalized[6:]}"
    ]
    
    try:
        with sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True) as conn:
            
            query = """
            SELECT 
                m.ROWID as message_id,
                m.text,
                m.attributedBody,
                m.is_from_me,
                datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch', 'localtime') as date_sent,
                m.date as timestamp,
                h.id as contact
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE ({}) 
            ORDER BY m.date ASC
            """.format(" OR ".join(["h.id LIKE ?" for _ in patterns]))
            
            cursor = conn.execute(query, patterns)
            messages = cursor.fetchall()
            
            console.print(f"📊 Found {len(messages)} raw messages")
            
            extracted_messages = []
            
            for row in track(messages, description="Decoding messages..."):
                message_id, text, attributed_body, is_from_me, date_sent, timestamp, contact = row
                
                # Extract actual text
                actual_text = None
                
                if text and text.strip():
                    actual_text = text.strip()
                elif attributed_body:
                    decoded = decode_attributed_body(attributed_body)
                    if decoded and decoded.strip():
                        actual_text = decoded.strip()
                
                if actual_text:  # Only include messages with actual text
                    extracted_messages.append({
                        'message_id': message_id,
                        'text': actual_text,
                        'is_from_me': bool(is_from_me),
                        'date': date_sent,
                        'timestamp': timestamp,
                        'contact': contact
                    })
            
            console.print(f"✅ Extracted {len(extracted_messages)} messages with text")
            
            # Show stats
            your_msgs = [m for m in extracted_messages if m['is_from_me']]
            their_msgs = [m for m in extracted_messages if not m['is_from_me']]
            
            console.print(f"📈 Your messages: {len(your_msgs)}")
            console.print(f"📈 Their messages: {len(their_msgs)}")
            
            return extracted_messages
            
    except Exception as e:
        console.print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        console.print("Usage: python3 message_extractor.py <phone_number>")
        console.print("Example: python3 message_extractor.py 'XXXXXXXXXX'")
        sys.exit(1)
    
    phone = sys.argv[1]
    messages = extract_conversation(phone)
    
    if messages:
        # Save to file
        normalized = re.sub(r'[^\d]', '', phone)
        filename = f"conversation_{normalized}.json"
        with open(filename, 'w') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        
        console.print(f"💾 Saved to: {filename}")
        
        # Show samples
        console.print("\n💬 Recent messages:")
        for msg in messages[-5:]:
            sender = "You" if msg['is_from_me'] else "Them"
            preview = msg['text'][:60] + "..." if len(msg['text']) > 60 else msg['text']
            console.print(f"   {sender}: {preview}")
    else:
        console.print("❌ No messages found")