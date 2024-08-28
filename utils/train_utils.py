
import pandas as pd

def pred_sets(train_label, train_subjects, train_prediction, validation_label, validation_subjects,
              validation_prediction, test_subjects, test_label, test_prediction, network, ensemble_networks):
        
    predictions_df = pd.DataFrame(columns=['Subject', 'Real Label','Prediction', 'Network', 'Ensemble_networks', 'Set'])

    # Save the train predictions along with the real labels
    train_predictions_df = pd.DataFrame({
        'Subject': train_subjects,  # Assuming the subjects are sequentially numbered
        'Real Label': train_label,
        'Prediction': train_prediction,
        'Network': network,
        'Ensemble_networks': '_'.join(ensemble_networks),#args.ensemble_networks),
        'Set': 'Train'
    })
    # Save the validation and test predictions along with the real labels
    validation_predictions_df = pd.DataFrame({
        'Subject': validation_subjects,
        'Real Label': validation_label,
        'Prediction': validation_prediction,
        'Network': network,
        'Ensemble_networks': '_'.join(ensemble_networks),
        'Set' : 'Validation'
    })

    test_predictions_df = pd.DataFrame({
        'Subject': test_subjects,
        'Real Label': test_label,
        'Prediction': test_prediction,
        'Network': network,
        'Ensemble_networks': '_'.join(ensemble_networks),
        'Set' : 'Test'
    })

    predictions_df = pd.concat([predictions_df, validation_predictions_df, test_predictions_df, train_predictions_df])
    return predictions_df