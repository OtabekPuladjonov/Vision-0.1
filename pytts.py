import pyttsx3

engine = pyttsx3.init()
s={"one","two",'five'}
n_s=list(s)
for i in n_s:
    var=i
    engine.say(var)
engine.runAndWait()
