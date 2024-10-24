import re
import time

# Define intents with associated keywords and example phrases
intents = {
    'greeting': ['hello', 'hi', 'good morning', 'good afternoon', 'hey'],
    'hotels': ['hotel', 'stay', 'accommodation', 'rooms', 'region hotels', 'hotels in the region'],
    'restaurants': ['restaurant', 'food', 'eat', 'dine', 'dining'],
    'tour': ['tour', 'visit', 'sightseeing', 'places', 'landmarks'],
    'booking': ['booking', 'reserve', 'reservation', 'book'],
    'thanks': ['thanks', 'thank you', 'appreciate', 'grateful'],
    'goodbye': ['bye', 'goodbye', 'see you', 'farewell']
}

# Predefined responses for each intent
responses = {
    'greeting': "Hello! Welcome to Morano Calabro Travel. How can I assist you today?",
    'hotel': "You could stay at the Locanda del Parco, at the Meruo hotel, or at one of the many vacation homes available on MoranoCalabro.store",
    'restaurants': "Our favorite recommendation is the 'Antica Masseria Salmena, or the La Locanda del Parco'.",
    'tour': "We offer guided tours to Morano Calabro's historical landmarks. Would you like to book one?",
    'booking': "You can make a booking through this website : www.MoranoCalabro.travel",
    'thanks': "You're welcome! Let me know if there's anything else I can help with.",
    'goodbye': "Goodbye! Have a great day and enjoy your time in Morano Calabro!"
}

# Fallback response for unrecognized queries
fallback_response = "I'm sorry, I didn't quite understand that. Here are some things you can ask me about: hotels, restaurants, tours, or bookings."

# Function to match user input with keywords
def classify_intent(user_input):
    user_input = user_input.lower()
    matched_intents = []
    
    # Check for keyword matches in the user input
    for intent, keywords in intents.items():
        for keyword in keywords:
            if re.search(rf'\b{keyword}\b', user_input):  # Matches whole words or phrases
                matched_intents.append(intent)
    
    return matched_intents

# Function to handle multiple intents in one query
def handle_multiple_intents(matched_intents):
    if len(matched_intents) == 1:
        return responses[matched_intents[0]]
    elif len(matched_intents) > 1:
        # Ask the user which intent they prefer, or respond with both
        response = "It seems you're asking about multiple things: " + ', '.join(matched_intents) + ". "
        response += "Would you like to hear about " + matched_intents[0] + " or " + matched_intents[1] + " first?"
        return response
    else:
        return None

# Main function to run the chatbot
def chatbot():
    print("Bot: Hello! Welcome to Morano Calabro Travel. Ask me anything about hotels, tours, restaurants, or bookings.")
    
    while True:
        user_input = input("\nYou: ")
        
        # Exit the chatbot if the user says goodbye
        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("Bot: " + responses['goodbye'])
            break
        
        # Determine the intent of the user input
        matched_intents = classify_intent(user_input)
        
        # Respond based on the identified intent(s)
        if matched_intents:
            response = handle_multiple_intents(matched_intents)
            time.sleep(1)  # Simulate response delay for realism
            print("Bot: " + response)
        else:
            time.sleep(1)  # Simulate response delay
            print("Bot: " + fallback_response)

if __name__ == '__main__':
    chatbot()