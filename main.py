from backprop_runner import backprop_model_runner
# from ipc_runner import ipc_model_runner

def main():
    '''
    If you experience an error cannot import name 'model' from 'Backprop_model.model
    DO the following:
    1. Go to the file runner of what would you like to run.
    2. Uncomment the function call.
    3. Type in your command line: python ipc_runner.py or python backprop_runner.py
    '''
    # Backpropagation Model 🚀
    backprop_model_runner()

    # Incremental Predictive Coding Model 🚀
    # ipc_model_runner()

main()
