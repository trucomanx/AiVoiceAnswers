
from deep_consultation.core       import consult_with_deepchat
from deep_consultation.core_audio import speech_file_transcript_deepinfra

    
SYSTEM_PROMPT = """
You are an expert in many fields, a true guru. 
Your mission is to respond to any question asked by the user. 
If you do not know the answer, you must be honest and clearly state that you do not have it. 
Your response should be a short, concise paragraph.
Avoid trivial conversations, redundant answers, and idle chatter.
"""

def consultation_in_depth(system_data, msg):

    OUT=consult_with_deepchat(  system_data["base_url"],
                                system_data["api_key"],
                                system_data["model_llm"],
                                msg,
                                SYSTEM_PROMPT)
    return OUT

def transcription_in_depth(system_data, filepath, language=None):

    OUT=speech_file_transcript_deepinfra(   system_data["base_url"],
                                            system_data["api_key"],
                                            system_data["model_transcript"],
                                            filepath,
                                            language=language)
    return OUT    
