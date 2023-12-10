#!/bin/zsh

# Run the figure plotting scripts

######## These are commented out because they take a long time to run, uncomment them if you want to run them
######## to train the model, you need to use scikit-optimize, which plays poorly with numpy, a more granular environment is suggested.


#if ! poetry run python based/data_format.py; then
#    echo "Execution failed: based/data_format.py"
#    exit 1
#fi

#if ! poetry run python based/based_trainer_sans_trampush.py; then
#    echo "Execution failed: based/based_trainer_sans_trampush.py"
#    exit 1
#fi

#if ! poetry run python based/based_trainer_full.py; then
#    echo "Execution failed: based/based_trainer_full.py"
#    exit 1
#fi

########
########

if ! poetry run python based/based_validation.py; then
    echo "Execution failed: based/based_validation.py"
    exit 1
fi

if ! poetry run python figure_plotting/reproduce_figs.py; then
    echo "Execution failed: figure_plotting/reproduce_figs.py"
    exit 1
fi

if ! poetry run python figure_plotting/Figure1_histogram.py; then
    echo "Execution failed: figure_plotting/Figure1_histogram.py"
    exit 1
fi

if ! poetry run python figure_plotting/Figure2.py; then
    echo "Execution failed: figure_plotting/Figure2.py"
    exit 1
fi

if ! poetry run python figure_plotting/Figure3.py; then
    echo "Execution failed: figure_plotting/Figure3.py"
    exit 1
fi

if ! poetry run python figure_plotting/extended_data.py; then
    echo "Execution failed: figure_plotting/extended_data.py"
    exit 1
fi
