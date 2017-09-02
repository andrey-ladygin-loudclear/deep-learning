def rnn(cell, input_list, initial_state):
    state = initial_state
    ouputs = []
    for i, input in enumerate(input_list):
        ouput, state = cell(input, state)
        ouputs.append(ouput)
    return (ouputs, state)