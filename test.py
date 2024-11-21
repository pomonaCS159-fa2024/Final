import llama_stack
print(dir(llama_stack))
import speech_recognition as sr
import openai
import os

openai.api_key = os.getenv('API_KEY')

def convert_speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)

        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None

def generate_response_from_text(input_text):
    prompt = f"Given the following text, create a prompt for a response: \n{input_text}\n"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        generated_prompt = response.choices[0].text.strip()
        print(generated_prompt)
        return generated_prompt
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        return None

def main(audio_file_path):
    text_data = convert_speech_to_text(audio_file_path)
    if text_data:
        response_prompt = generate_response_from_text(text_data)
        return response_prompt
    else:
        return None

if __name__ == "__main__":
    audio_file_path = "audiorec2.wav"
    main(audio_file_path)
