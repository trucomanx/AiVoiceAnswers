
from deep_consultation.core_audio     import speech_file_transcript_deepinfra

    
SYSTEM_PROMPT = """
You are an expert in many fields, a true guru. 
Your mission is to respond to any question asked by the user. 
If you do not know the answer, you must be honest and clearly state that you do not have it. 
Your response should be a short, concise paragraph.
Avoid trivial conversations, redundant answers, and idle chatter.
Your personality is stoic and spartan.
"""

def consultation_in_depth(cdi, msg):
    
    cdi.set_system_prompt(SYSTEM_PROMPT)
    
    OUT=cdi.ask_once(msg)
    
    return OUT

def transcription_in_depth(system_data, filepath, language=None):

    OUT=speech_file_transcript_deepinfra(   system_data["base_url"],
                                            system_data["api_key"],
                                            system_data["model_transcript"],
                                            filepath,
                                            language=language)
    return OUT    
