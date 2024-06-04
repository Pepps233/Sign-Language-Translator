import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 130)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.say("")
engine.runAndWait()
