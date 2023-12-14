import warnings
import pandas as pd
import xgboost as xgb
import sys
sys.path.append('../codes')
import utils as utils
import gradio as gr
import process_features
import deconstruct_sigs_from_user_input as deconstructSigs

xgb_onconpc = xgb.Booster()
xgb_onconpc.load_model('../model/xgboost_v1.7.6_OncoNPC_full.json')
cancer_types_to_consider = ['Acute Myeloid Leukemia', 'Bladder Urothelial Carcinoma', 'Cholangiocarcinoma',
                            'Colorectal Adenocarcinoma', 'Diffuse Glioma', 'Endometrial Carcinoma',
                            'Esophagogastric Adenocarcinoma', 'Gastrointestinal Neuroendocrine Tumors', 'Gastrointestinal Stromal Tumor',
                            'Head and Neck Squamous Cell Carcinoma',
                            'Invasive Breast Carcinoma', 'Melanoma', 'Meningothelial Tumor',
                            'Non-Hodgkin Lymphoma', 'Non-Small Cell Lung Cancer', 'Ovarian Epithelial Tumor', 'Pancreatic Adenocarcinoma',
                            'Pancreatic Neuroendocrine Tumor', 'Pleural Mesothelioma', 'Prostate Adenocarcinoma', 'Renal Cell Carcinoma',
                            'Well-Differentiated Thyroid Cancer']   

all_features = pd.read_csv('../data/onconpc_features.csv').drop('Unnamed: 0', axis=1).columns.tolist()

def get_preds(patients_file, samples_file, mutations_file, cna_file, tumor_id):

    """
    Generates predictions and explanations for given tumor samples using OncoNPC model.

    This function processes patient, sample, mutation, and CNA data to predict primary sites of 
    Cancer of Unknown Primary (CUP) tumors. It also provides a bar chart of SHAP values to explain the predictions.

    Args:
        patients_file: A csv file object representing patient data.
        samples_file: A csv file object representing sample data.
        mutations_file: A csv file object representing mutation data.
        cna_file: A csv file object representing CNA (Copy Number Alterations) data.
        tumor_id: The ID of the tumor.

    Returns:
        A tuple containing:
            A string containing the top 3 most probable cancers along with their predicted probabilities. 
            The filepath to the SHAP value bar chart explaining the prediction for the given tumor ID.
    """
    
    # convert files to data frames
    patients_df = pd.read_csv(patients_file.name, sep='\t')
    samples_df = pd.read_csv(samples_file.name, sep='\t')
    mutations_df = pd.read_csv(mutations_file.name, sep='\t')
    cna_df = pd.read_csv(cna_file.name, sep='\t') 


    # declared as global variables to generate plots in update_image function
    global sample_id 
    global features
    global predictions

    # get features and labels for OncoNPC predictive inference
    df_features_genie_final, df_labels_genie = utils.get_onconpc_features_from_raw_data(
        patients_df,
        samples_df,
        mutations_df,
        cna_df,
        features_onconpc_path='../data/features_onconpc.pkl',
        combined_cohort_age_stats_path='../data/combined_cohort_age_stats.pkl',
        mut_sig_weights_filepath='../data/mutation_signatures/sigProfiler*.csv'
    )
    df_features_genie_final.to_csv('../data/onconpc_features2.csv')

    sample_id = tumor_id
    features = df_features_genie_final

    # load fully trained OncoNPC model
    xgb_onconpc = xgb.Booster()
    xgb_onconpc.load_model('../model/xgboost_v1.7.6_OncoNPC_full.json')
    
    # predict primary sites of CUP tumors
    predictions = utils.get_xgboost_latest_cancer_type_preds(xgb_onconpc,
                                                          df_features_genie_final,
                                                          cancer_types_to_consider)

    # get SHAP values for CUP tumors
    warnings.filterwarnings('ignore')
    shaps = utils.obtain_shap_values_with_latest_xgboost(xgb_onconpc, df_features_genie_final)
    

    query_ids = list(samples_df.SAMPLE_ID.values)

    # results is structured such that:
    # results_dict[query_id] = {'pred_prob': pred_prob,'pred_cancer': pred_cancer,'explanation_plot': full_filename}
    results = utils.get_onconpc_prediction_explanations(query_ids, predictions, shaps,
                                                        df_features_genie_final,
                                                        cancer_types_to_consider,
                                                        save_plot=True) 

    return get_top3(predictions, tumor_id), 'category explanation', results[tumor_id]['explanation_plot'] + '.png' 

def parse_inputs(age, gender, CNA_events, mutations):
    # Normalization of age input is to be done later
    age = age # TODO: Check how this is normalized 

    # Convert gender to numerical value: male as 1, otherwise -1
    gender = 1 if gender == 'male' else -1 
    
    # Process CNA events: Expected format [[CNA, val], [CNA, val]]
    if len(CNA_events) > 0:
        CNA_events = CNA_events.split('|')
        for i in range(len(CNA_events)):
            # Split each event into CNA and value, and cast the value to integer
            CNA, val = CNA_events[i].split()
            CNA_events[i] = [CNA + ' CNA', int(val)] # Cast val to integer
    else:
        CNA_events = []

    # Process mutations: Expected format [mut1, mut2, etc.]
    if len(mutations) > 0:
        mutations = mutations.split('| ')
        for i in range(len(mutations)):
            # Split each mutation entry and strip white space
            mutations[i] = ['manual input'] + mutations[i].split(', ')
            mutations[i] = [m.strip() for m in mutations[i]] # Strip white space
    else:
        mutations = []

    # Define mutation columns for DataFrame and create mutation DataFrame
    mutation_columns = ["UNIQUE_SAMPLE_ID", "CHROMOSOME", "POSITION", "REF_ALLELE", "ALT_ALLELE"]
    mutation_df = pd.DataFrame(mutations, columns=mutation_columns) if mutations else pd.DataFrame(columns=mutation_columns)

    # Save mutation data to a CSV file
    mutation_df.to_csv('./mutation_input.csv', index=False)

    # Get base substitution file and read into DataFrame
    base_sub_file = deconstructSigs.get_base_substitutions() 
    df_trinuc_feats = pd.read_csv(base_sub_file) 
    # Obtain mutation signatures
    mutation_signatures = process_features.obtain_mutation_signatures(df_trinuc_feats)    

    # Initialize data dictionary and populate with age and mutation signatures
    data = {'Age': age} # TODO: Check how this is normalized 
    for column in mutation_signatures.columns:
        data[column] = mutation_signatures.loc[0][column]
    
    # Add CNA events to data dictionary
    for CNA, val in CNA_events:
        data[CNA] = val
        
    # Add zero values for missing features in data dictionary
    for column in all_features:
        if column not in data.keys():
            data[column] = 0

    # Return the data as a DataFrame
    return pd.DataFrame([data])

import pandas as pd
import utils  # Assuming utils is a module with relevant functions

def get_preds_min_info(age, gender, CNA_events, mutations, output='Top Prediction'):
    """
    Generate predictions and explanations for cancer type based on input features.

    Parameters:
    age (int or float): The age of the individual.
    gender (str): The gender of the individual, either 'male' or 'female'.
    CNA_events (str): A string of CNA events, formatted appropriately.
    mutations (str): A string of mutation data, formatted appropriately.
    output (str, optional): Specifies the type of output; default is 'Top Prediction'.

    Returns:
    tuple: A tuple containing top 3 predictions and the path to the explanation plot.
    """
    global sample_id
    global features
    global predictions

    # Parse input features
    features = parse_inputs(age, gender, CNA_events, mutations)
    
    # Generate predictions using the XGBoost model
    predictions = pd.DataFrame(utils.get_xgboost_latest_cancer_type_preds(xgb_onconpc, features, cancer_types_to_consider))
    
    # Compute SHAP values for model explanation
    shaps = utils.obtain_shap_values_with_latest_xgboost(xgb_onconpc, features)


    # Assuming a single sample is being processed
    query_ids = [0]
    sample_id = 0
    
    # Generate explanations for the predictions
    results = utils.get_onconpc_prediction_explanations(query_ids, predictions, shaps,
                                                        features, cancer_types_to_consider,
                                                        save_plot=True)

    # Return the top 3 predictions and the path to the explanation plot
    return get_top3(predictions, 0), 'category explanation', results[0]['explanation_plot'] + '.png'

def get_top3(predictions, tumor_sample_id):
    """
    Extracts and formats the top three cancer type predictions for a given tumor sample.

    Args:
        predictions: A DataFrame containing the cancer type predictions for various samples.
        tumor_sample_id: The ID of the tumor sample for which to extract the top three predictions.

    Returns:
        A string that lists the top three predicted cancer types and their probabilities.
    """
    # select the row corresponding to the tumor sample ID
    result = predictions.loc[tumor_sample_id]

    # transpose the row for easier processing, each row has columns cancer type, cancer probability 
    transposed_row = result.transpose()

    # remove unnecessary rows
    transposed_row = transposed_row.drop(['cancer_type', 'max_posterior'])

    # convert the series to a DataFrame and rename the column
    transposed_row = transposed_row.to_frame()
    transposed_row.columns = ['probability']

    # make sure the probability column is numeric
    transposed_row['probability'] = pd.to_numeric(transposed_row['probability'], errors='coerce')

    # get the top 3 predictions and their probabilities
    top3df = transposed_row.nlargest(3, columns=['probability'])
    top3 = top3df.index.tolist() # cancer types are indices
    top3probs = top3df['probability'].tolist()

    # build a formatted string with the top 3 predictions
    build = ''
    for cancer, prob in zip(top3, top3probs):
        build += f'{cancer}: {prob:.2f}\n'
    build = build.rstrip('\n')

    return build

import gradio as gr

global image # path to explanation plot, defined as global for the purposes of update 
global features

def extract_sample_ids(samples_file):
    # Read the file into a DataFrame
    if samples_file is None:
        return []

    df = pd.read_csv(samples_file.name, sep='\t')
    # Assuming the column containing the sample IDs is named 'SampleID'
    sample_ids = df['SAMPLE_ID'].unique().tolist()
    return  gr.Dropdown.update(choices=sample_ids)

def update_image(target):
    global image
    global features
    global predictions
    
    shaps_cup = utils.obtain_shap_values_with_latest_xgboost(xgb_onconpc, features) # get shap values 

    target_idx = cancer_types_to_consider.index(target) # index of cancer type prediction 
    
    # Get SHAP-based explanation for the prediction
    feature_sample_df = features.loc[sample_id] # find the exact tumor sample we're predicting for 
    shap_pred_cancer_df = pd.DataFrame(shaps_cup[target_idx],
                                       index=features.index,
                                       columns=features.columns)
    shap_pred_sample_df = shap_pred_cancer_df.loc[sample_id]
    probability = predictions.loc[sample_id][target]
    
    # Generate explanation plot
    sample_info = f'Prediction: {target}\nPrediction probability: {probability:.3f}'
    feature_group_to_features_dict, feature_to_feature_group_dict = utils.partition_feature_names_by_group(features.columns)
    fig = utils.get_individual_pred_interpretation(shap_pred_sample_df, feature_sample_df, feature_group_to_features_dict, feature_to_feature_group_dict,sample_info=sample_info, filename=f'{target}_plot.png', filepath='../others_prediction_explanation', save_plot=True)
    return fig

def show_row(value):
    if value=="CSV File":
        return (gr.update(visible=True), gr.update(visible=False))  
    if value=="Manual Inputs":
        return (gr.update(visible=False), gr.update(visible=True))
    return (gr.update(visible=False), gr.update(visible=False))

def launch_gradio(server_name, server_port):
    cancer_types_to_consider = ['Acute Myeloid Leukemia', 'Bladder Urothelial Carcinoma', 'Cholangiocarcinoma',
                                'Colorectal Adenocarcinoma', 'Diffuse Glioma', 'Endometrial Carcinoma',
                                'Esophagogastric Adenocarcinoma', 'Gastrointestinal Neuroendocrine Tumors', 'Gastrointestinal Stromal Tumor',
                                'Head and Neck Squamous Cell Carcinoma',
                                'Invasive Breast Carcinoma', 'Melanoma', 'Meningothelial Tumor',
                                'Non-Hodgkin Lymphoma', 'Non-Small Cell Lung Cancer', 'Ovarian Epithelial Tumor', 'Pancreatic Adenocarcinoma',
                                'Pancreatic Neuroendocrine Tumor', 'Pleural Mesothelioma', 'Prostate Adenocarcinoma', 'Renal Cell Carcinoma',
                                'Well-Differentiated Thyroid Cancer']   

    with gr.Blocks() as demo:
        d = gr.Dropdown(["Manual Inputs", "CSV File"])
        with gr.Row(visible=False) as r1:
            with gr.Column():
                patients_file = gr.File(label="Upload clinical patients data")
                samples_file = gr.File(label="Upload clinical samples data")
                mutations_file = gr.File(label="Upload mutations data")
                cna_file = gr.File(label="Upload CNA data")
                tumor_sample_id = gr.Dropdown(choices=[], label="Tumor Sample ID")  # Changed to Dropdown
                submit_button = gr.Button("Submit")

            with gr.Column():
                predictions_output = gr.Textbox(label="Top 3 Predicted Cancer Types")
                category_explanation = gr.Textbox(label='Feature Category Explanation', interactive='False')
                image = gr.Image(label="Image Display")
                output_selector = gr.Dropdown(choices=cancer_types_to_consider, label="Output Options", filterable=True)

            samples_file.change(extract_sample_ids, inputs=samples_file, outputs=tumor_sample_id)
            submit_button.click(get_preds, inputs=[patients_file, samples_file, mutations_file, cna_file, tumor_sample_id], outputs=[predictions_output, category_explanation, image])
            output_selector.change(update_image, inputs=output_selector, outputs=image)

        with gr.Row(visible=False) as r2:
            with gr.Column():
                age = gr.Number(label="Age")
                gender = gr.Radio(choices=["Male", "Female"], label="Gender")
                cna_events = gr.Textbox(lines=2, placeholder="Enter CNA events...\nex: KCNQ1 2 | BRAF -1 | SLX1B 1 | CBLB -2", label="Genes with CNA Events (comma-separated)")
                mutations = gr.Textbox(lines=5, placeholder="Enter mutations...\nex: chr17, 7577539, G, A | chr3, 178936091, G, A | chr6, 152419920, T, A", label="MUTATIONS")
                submit_button = gr.Button("Submit")
            with gr.Column():
                predictions_output = gr.Textbox(label="Top 3 Predicted Cancer Types")
                category_explanation = gr.Textbox(label='Feature Category Explanation', interactive='False')
                image = gr.Image(label="Image Display") 
                output_selector = gr.Dropdown(choices=cancer_types_to_consider, label="Output Options", filterable=True)
            submit_button.click(get_preds_min_info, inputs=[age, gender, cna_events, mutations], outputs=[predictions_output, category_explanation, image])
            output_selector.change(update_image, inputs=output_selector, outputs=image)

        d.change(show_row, d, [r1, r2])
        
    demo.launch(debug=True, share=True,server_name=server_name, server_port=server_port)