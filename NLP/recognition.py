import speech_recognition as sr
from transformers import Conversation, pipeline
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

conversational_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")

def speak(text):
    engine.say(text)
    engine.runAndWait()

with sr.Microphone() as source:
    print("Listening...")

    recognizer.adjust_for_ambient_noise(source)

    while True:
        audio_data = recognizer.listen(source)
        converse = True
        try:
            print("Recognizing...")

            # Recognize the audio and convert it to text
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)

            # Check if the phrase "Hey Sushi" is said
            if "hey sushi" in text.lower():
                print("Activation phrase detected. Starting conversation...")
                text = text.lower()
                text = text.split("hey sushi ")
                text = text[1]
                print(text)
                # Continue with the conversation pipeline
                while converse:
                    if "end hearing" in text.lower():
                        print("Ending listening...")
                        break
                    
                    conversation = Conversation(eos_token_id=50256)
                    conversation.add_user_input(text)
                    response = conversational_pipeline([conversation])
                    generated_text = response[1]["content"]
                    print("Assistant:", generated_text)

                    # Speak the response
                    speak(generated_text)
                    converse = False
            else:
                print("Activation phrase not detected. Listening for 'Hey Sushi'")

        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
