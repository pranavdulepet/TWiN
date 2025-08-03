#!/usr/bin/env python3
"""
Message Analysis Engine - TextTwin Project
==========================================

Analyzes message patterns to understand personal texting style.
Works with any message data source (CSV, JSON, or database).
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import textstat
import emoji
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@dataclass
class MessageStyle:
    """Data class to hold analyzed message style patterns."""
    avg_message_length: float
    avg_words_per_message: float
    avg_sentences_per_message: float
    punctuation_habits: Dict[str, float]
    capitalization_ratio: float
    emoji_usage_rate: float
    common_phrases: List[Tuple[str, int]]
    response_time_patterns: Dict[str, float]
    conversation_starters: List[str]
    unique_vocabulary: int
    readability_score: float
    sentiment_indicators: Dict[str, int]
    time_of_day_patterns: Dict[str, int]

class MessageAnalyzer:
    """Analyzes message patterns to create a personal texting style profile."""
    
    def __init__(self):
        self.messages_df = None
        self.style_profile = None
        
    def load_from_csv(self, csv_path: str) -> None:
        """Load messages from CSV file."""
        self.messages_df = pd.read_csv(csv_path)
        console.print(f"✓ Loaded {len(self.messages_df)} messages from CSV")
        
    def load_from_sample_data(self) -> None:
        """Create sample data for testing when real iMessage data isn't available."""
        sample_messages = [
            {"text": "hey what's up? 😊", "timestamp": "2024-01-01 10:00:00", "is_from_me": True},
            {"text": "not much, just chillin. you?", "timestamp": "2024-01-01 15:30:00", "is_from_me": True},
            {"text": "lol same here... working on some code", "timestamp": "2024-01-01 16:45:00", "is_from_me": True},
            {"text": "btw did you see that new movie?", "timestamp": "2024-01-01 18:20:00", "is_from_me": True},
            {"text": "nah not yet! is it good??", "timestamp": "2024-01-01 19:10:00", "is_from_me": True},
            {"text": "yeah it's pretty decent, worth watching", "timestamp": "2024-01-01 20:00:00", "is_from_me": True},
            {"text": "cool I'll check it out this weekend", "timestamp": "2024-01-02 09:15:00", "is_from_me": True},
            {"text": "awesome! let me know what you think", "timestamp": "2024-01-02 11:30:00", "is_from_me": True},
            {"text": "will do 👍", "timestamp": "2024-01-02 12:00:00", "is_from_me": True},
            {"text": "hey are we still on for tonight?", "timestamp": "2024-01-02 17:45:00", "is_from_me": True},
            {"text": "omg totally forgot!! yes definitely", "timestamp": "2024-01-02 18:30:00", "is_from_me": True},
            {"text": "haha no worries, see you at 7?", "timestamp": "2024-01-02 18:35:00", "is_from_me": True},
            {"text": "perfect! can't wait 🎉", "timestamp": "2024-01-02 18:40:00", "is_from_me": True},
        ]
        
        self.messages_df = pd.DataFrame(sample_messages)
        self.messages_df['timestamp'] = pd.to_datetime(self.messages_df['timestamp'])
        console.print(f"✓ Created sample dataset with {len(self.messages_df)} messages")
        
    def analyze_style(self) -> MessageStyle:
        """Perform comprehensive style analysis."""
        if self.messages_df is None:
            raise ValueError("No messages loaded. Use load_from_csv() or load_from_sample_data() first.")
        
        # Filter to only user messages
        user_messages = self.messages_df[self.messages_df['is_from_me'] == True]['text'].tolist()
        
        console.print("[bold blue]Analyzing message style patterns...[/bold blue]")
        
        # Basic message metrics
        message_lengths = [len(msg) for msg in user_messages]
        word_counts = [len(msg.split()) for msg in user_messages]
        sentence_counts = [len(re.split(r'[.!?]+', msg)) - 1 for msg in user_messages]
        
        # Punctuation analysis
        punctuation_counts = defaultdict(int)
        total_chars = 0
        for msg in user_messages:
            for char in msg:
                if char in '.,!?;:':
                    punctuation_counts[char] += 1
                total_chars += 1
        
        punctuation_habits = {p: count/len(user_messages) for p, count in punctuation_counts.items()}
        
        # Capitalization analysis
        capital_chars = sum(1 for msg in user_messages for char in msg if char.isupper())
        total_letters = sum(1 for msg in user_messages for char in msg if char.isalpha())
        capitalization_ratio = capital_chars / max(total_letters, 1)
        
        # Emoji analysis
        emoji_count = sum(len(emoji.emoji_list(msg)) for msg in user_messages)
        emoji_usage_rate = emoji_count / len(user_messages)
        
        # Common phrases (2-3 word combinations)
        all_text = ' '.join(user_messages).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        # Bigrams and trigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        common_phrases = Counter(bigrams + trigrams).most_common(10)
        
        # Conversation starters
        starters = []
        for msg in user_messages:
            if len(msg.split()) <= 5:  # Short messages likely to be conversation starters
                first_words = ' '.join(msg.split()[:2]).lower()
                if first_words not in starters:
                    starters.append(first_words)
        
        # Vocabulary analysis
        unique_words = set(re.findall(r'\b\w+\b', all_text.lower()))
        
        # Readability (average)
        readability_scores = []
        for msg in user_messages:
            if len(msg.strip()) > 10:  # Only analyze substantial messages
                try:
                    score = textstat.flesch_reading_ease(msg)
                    readability_scores.append(score)
                except:
                    pass
        
        avg_readability = np.mean(readability_scores) if readability_scores else 50.0
        
        # Sentiment indicators (simple keyword analysis)
        positive_words = ['good', 'great', 'awesome', 'cool', 'nice', 'love', 'like', 'happy', 'yes', 'lol', 'haha']
        negative_words = ['bad', 'terrible', 'hate', 'no', 'ugh', 'annoying', 'stupid', 'worst']
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        
        sentiment_counts = {
            'positive': sum(1 for word in positive_words if word in all_text),
            'negative': sum(1 for word in negative_words if word in all_text),
            'questions': sum(1 for word in question_words if word in all_text)
        }
        
        # Time patterns (if timestamp available)
        time_patterns = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
        if 'timestamp' in self.messages_df.columns:
            user_msgs_df = self.messages_df[self.messages_df['is_from_me'] == True]
            for _, row in user_msgs_df.iterrows():
                hour = pd.to_datetime(row['timestamp']).hour
                if 6 <= hour < 12:
                    time_patterns['morning'] += 1
                elif 12 <= hour < 17:
                    time_patterns['afternoon'] += 1
                elif 17 <= hour < 22:
                    time_patterns['evening'] += 1
                else:
                    time_patterns['night'] += 1
        
        # Create style profile
        self.style_profile = MessageStyle(
            avg_message_length=np.mean(message_lengths),
            avg_words_per_message=np.mean(word_counts),
            avg_sentences_per_message=np.mean(sentence_counts),
            punctuation_habits=punctuation_habits,
            capitalization_ratio=capitalization_ratio,
            emoji_usage_rate=emoji_usage_rate,
            common_phrases=common_phrases,
            response_time_patterns={},  # Would need conversation context
            conversation_starters=starters[:10],
            unique_vocabulary=len(unique_words),
            readability_score=avg_readability,
            sentiment_indicators=sentiment_counts,
            time_of_day_patterns=time_patterns
        )
        
        console.print("✓ Style analysis complete!")
        return self.style_profile
    
    def display_style_report(self) -> None:
        """Display a comprehensive style analysis report."""
        if not self.style_profile:
            console.print("[red]No style profile available. Run analyze_style() first.[/red]")
            return
        
        # Main statistics table
        table = Table(title="📱 Your Texting Style Profile", style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Average message length", f"{self.style_profile.avg_message_length:.1f} characters")
        table.add_row("Average words per message", f"{self.style_profile.avg_words_per_message:.1f}")
        table.add_row("Capitalization ratio", f"{self.style_profile.capitalization_ratio:.2%}")
        table.add_row("Emoji usage rate", f"{self.style_profile.emoji_usage_rate:.2f} per message")
        table.add_row("Unique vocabulary size", f"{self.style_profile.unique_vocabulary} words")
        table.add_row("Readability score", f"{self.style_profile.readability_score:.1f}/100")
        
        console.print(table)
        
        # Punctuation habits
        if self.style_profile.punctuation_habits:
            punct_table = Table(title="Punctuation Habits", style="green")
            punct_table.add_column("Punctuation", style="bold")
            punct_table.add_column("Usage per message", justify="right")
            
            for punct, rate in sorted(self.style_profile.punctuation_habits.items(), 
                                    key=lambda x: x[1], reverse=True):
                punct_table.add_row(punct, f"{rate:.2f}")
            
            console.print(punct_table)
        
        # Common phrases
        if self.style_profile.common_phrases:
            phrases_table = Table(title="Your Most Common Phrases", style="yellow")
            phrases_table.add_column("Phrase", style="bold")
            phrases_table.add_column("Count", justify="right")
            
            for phrase, count in self.style_profile.common_phrases[:8]:
                phrases_table.add_row(phrase, str(count))
            
            console.print(phrases_table)
        
        # Time patterns
        if any(self.style_profile.time_of_day_patterns.values()):
            time_table = Table(title="When You Text Most", style="magenta")
            time_table.add_column("Time of Day", style="bold")
            time_table.add_column("Message Count", justify="right")
            
            for time_period, count in self.style_profile.time_of_day_patterns.items():
                time_table.add_row(time_period.title(), str(count))
            
            console.print(time_table)
        
        # Conversation starters
        if self.style_profile.conversation_starters:
            starters_panel = Panel(
                "\n".join(f"• {starter}" for starter in self.style_profile.conversation_starters[:6]),
                title="Your Typical Conversation Starters",
                style="blue"
            )
            console.print(starters_panel)
    
    def save_style_profile(self, filepath: str) -> None:
        """Save the style profile to a JSON file."""
        if not self.style_profile:
            raise ValueError("No style profile to save. Run analyze_style() first.")
        
        # Convert dataclass to dict for JSON serialization
        profile_dict = {
            'avg_message_length': self.style_profile.avg_message_length,
            'avg_words_per_message': self.style_profile.avg_words_per_message,
            'avg_sentences_per_message': self.style_profile.avg_sentences_per_message,
            'punctuation_habits': self.style_profile.punctuation_habits,
            'capitalization_ratio': self.style_profile.capitalization_ratio,
            'emoji_usage_rate': self.style_profile.emoji_usage_rate,
            'common_phrases': self.style_profile.common_phrases,
            'conversation_starters': self.style_profile.conversation_starters,
            'unique_vocabulary': self.style_profile.unique_vocabulary,
            'readability_score': self.style_profile.readability_score,
            'sentiment_indicators': self.style_profile.sentiment_indicators,
            'time_of_day_patterns': self.style_profile.time_of_day_patterns,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        console.print(f"✓ Style profile saved to {filepath}")


def demo_analyzer():
    """Demonstrate the message analyzer with sample data."""
    analyzer = MessageAnalyzer()
    
    # Load sample data (since we don't have access to real iMessage data)
    analyzer.load_from_sample_data()
    
    # Analyze the style
    analyzer.analyze_style()
    
    # Display the report
    analyzer.display_style_report()
    
    # Save the profile
    analyzer.save_style_profile('my_texting_style.json')


if __name__ == "__main__":
    demo_analyzer()