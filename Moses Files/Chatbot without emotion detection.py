import random
import re
from collections import defaultdict
from textblob import TextBlob
from rake_nltk import Rake

class InteractiveConversationalBot:
    def __init__(self):
        self.memory = defaultdict(list)  # To remember conversation context
        self.user_name = None
        self.default_responses = [
            "Could you tell me more about that?",
            "That sounds interesting! What else?",
            "And then what happened?",
            "Do you often think about this?",
            "How does that make you feel?"
        ]
        self.engagement_questions = [
            "What do you think about this?",
            "How does this make you feel?",
            "Can you share more details?",
            "What are your thoughts on this?"
        ]
    
    def extract_keywords(self, text):
        # Basic keyword extraction: remove stopwords and simple text processing
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lower the case
        # Initialize RAKE
        rake = Rake()


        # Extract keywords
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()
        return keywords
    
    def generate_followup_question(self, keyword):
        # Simple dynamic question generation based on keywords
        questions = [
            
            f"What do you think about {keyword}?",
            f"How do you feel about {keyword}?",
            f"Can you elaborate more on {keyword}?",
            f"Do you often deal with {keyword}?"
        ]
        return random.choice(questions)
    
    def analyze_sentiment(self, text):
        # Analyze the sentiment of the user input using TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def personalize_response(self, text):
        # Adjust responses based on sentiment
        sentiment = self.analyze_sentiment(text)
        if sentiment > 0.5:
            return "I'm glad to hear that!"
        elif sentiment < -0.5:
            return "I'm sorry to hear that. Do you want to talk more about it?"
        else:
            return random.choice(self.engagement_questions)
    
    def get_response(self, user_input):
        if not self.user_name:
            self.user_name = user_input  # Capture the user's name
            return f"Nice to meet you, {self.user_name}! What would you like to talk about today?"
        
        keywords = self.extract_keywords(user_input)
        
        if keywords:
            keyword = random.choice(keywords)
            self.memory[keyword].append(user_input)
            followup = self.generate_followup_question(keyword)
            return f"{self.personalize_response(user_input)} {followup}"
        
        return random.choice(self.default_responses)

    def chat(self):
        print("Bot: Hi there! What's your name?")
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print(f"Bot: Goodbye, {self.user_name}! Have a great day!")
                break
            response = self.get_response(user_input)
            
            print(f"Bot: {response}")
            
# Start the conversational bot
bot = InteractiveConversationalBot()
bot.chat()
