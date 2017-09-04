"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = '../../deep-learning-master/tv-script-generation/data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
print(text)