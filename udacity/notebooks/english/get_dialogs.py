from bs4 import BeautifulSoup

with open('sentences.html') as f:
    lines = f.read()

dialogs = []

lines = BeautifulSoup(lines, "lxml").text.replace("\n\n\n", "\n").replace("\n\n", "\n").split("\n")

for i in range(len(lines)):
    print(lines[i])


for i in range(len(lines)):

    try:
        question_identificator = lines[i]
        question = lines[i + 1]
        answer_identificator = lines[i + 2]
        if len(lines[i + 3]) < 3:
            answer = lines[i + 4]
        else:
            answer = lines[i + 3]
    except IndexError:
        continue

    if 'Q.' in question_identificator and 'A.' in answer_identificator:

        print('question_identificator', question_identificator)
        print('question', question)
        print('answer_identificator', answer_identificator)
        print('answer', answer)
        print('----------------------------')
        dialogs.append({
            'Q': question,
            'A': answer,
        })


print(dialogs)
for raw in dialogs:
    line = raw['Q'] + "\n"
    line += raw['A'] + "\n"
    line += "-----------" + "\n"
    print(line)
    print(line, file='dialogs.txt')

#print(len(dialogs))