
from deep_consultation.core_audio     import speech_file_transcript_deepinfra

def transcription_in_depth(system_data, filepath, language=None):

    OUT=speech_file_transcript_deepinfra(   system_data["base_url"],
                                            system_data["api_key"],
                                            system_data["model_transcript"],
                                            filepath,
                                            language=language)
    return OUT    
