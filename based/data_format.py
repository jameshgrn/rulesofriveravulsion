import pandas as pd

def generate_data():
    """
    Generates a consolidated dataset from multiple sources, including the Dunne Jerolmack Dataset, Deal Dataset,
    and Fulton Dataset. The data is cleaned, transformed, and concatenated into a single DataFrame. The resulting
    DataFrame is then saved to a CSV file named 'based_input_data.csv'.

    The specific data transformations and cleaning steps include:
    - Loading datasets from Excel and CSV files.
    - Renaming columns and converting data types.
    - Dropping unnecessary columns.
    - Handling units conversions.
    - Combining and concatenating data from different sources.
    - Removing rows with missing values in key columns.

    The generated dataset is suitable for further analysis and modeling in hydrology-related research.

    Usage:
        Call this function to generate and save the consolidated dataset.

    """
    # Load Dunne Jerolmack Dataset
    dunne_jerolmack_path = r"data/BASED_model/GlobalDatasets.xlsx"
    dunne_jerolmack = pd.read_excel(dunne_jerolmack_path)

    # Drop unnecessary columns
    dunne_jerolmack = dunne_jerolmack.drop(['tau_*bf', 'D50 (m)'], axis=1)

    # Add a 'bankfull' column with True values as they are all bankfull (Dunne and Jerolmack, 2018)
    dunne_jerolmack['bankfull'] = True

    # Rename columns for consistency
    dunne_jerolmack.columns = ['source', 'site_id', 'slope', 'width', 'depth', 'discharge', 'bankfull']

    # Convert 'source' and 'site_id' to strings
    dunne_jerolmack['source'] = dunne_jerolmack['source'].astype(str)
    dunne_jerolmack['site_id'] = dunne_jerolmack['source'].astype(str)

    # Make sure 'slope' values are positive
    dunne_jerolmack['slope'] = dunne_jerolmack['slope'].abs()

    # Remove rows where 'source' contains 'Singer' case-insensitively as they are in the Deal Dataset
    dunne_jerolmack = dunne_jerolmack[~dunne_jerolmack['source'].str.contains('Singer', case=False, na=False)]

    # Load Deal Dataset
    deal_ds_path = "data/BASED_model/HG_data_comp_complete.csv"
    deal_ds = pd.read_csv(deal_ds_path)

    # Remove rows where 'river_class' is -1.0
    deal_ds = deal_ds.query("river_class != -1.0")

    # Drop unnecessary columns
    deal_ds = deal_ds.drop(['notes', 'area', 'sed_discharge', 'd90', 'bedload_discharge', 'erosion_rate', 'velocity',
                            'd50', 'd84', 'Unnamed: 0', 'DOI', 'primary_source', 'river_class'], axis=1)

    # Convert 'source' and 'site_id' to strings
    deal_ds['source'] = deal_ds['source'].astype(str)
    deal_ds['site_id'] = deal_ds['source'].astype(str)

    # Concatenate all dataframes
    based_input_data = pd.concat([deal_ds, dunne_jerolmack], axis=0)

    # Drop rows with missing values in key columns
    based_input_data = based_input_data.dropna(subset=['width', 'slope', 'discharge', 'depth'])

    # Save the resulting dataframe to a CSV file
    based_input_data.to_csv('data/based_input_data.csv')


if __name__ == '__main__':
    generate_data()
