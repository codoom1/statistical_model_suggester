"""Utilities to process synthetic data for plot generation."""
import pandas as pd
import numpy as np
import json
import re

def extract_data_from_r_output(text_output):
    """Extract data from R output text.
    
    Args:
        text_output: Text output from R code
        
    Returns:
        DataFrame or dict with extracted data
    """
    # Try to extract data frame-like structures
    # Look for patterns like:
    #   x1        x2        y
    # 1.23     4.56     7.89
    lines = text_output.split('\n')
    data_lines = []
    header = None
    
    in_data_section = False
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Look for table headers (usually have spaces and no > prefix)
        if not line.startswith('>') and ' ' in line and not header:
            # Check if it looks like a header
            possible_header = re.sub(r'\s+', ' ', line).strip()
            if re.match(r'^[A-Za-z0-9_.]+([ ]+[A-Za-z0-9_.]+)+$', possible_header):
                header = [col for col in re.split(r'\s+', possible_header) if col]
                in_data_section = True
                continue
                
        # If we're in a data section and line doesn't start with >, it might be data
        if in_data_section and not line.startswith('>'):
            # Clean the line
            clean_line = re.sub(r'\s+', ' ', line).strip()
            # Check if it's all numbers with possible row names
            if re.match(r'^([A-Za-z0-9_.]+ )?[-0-9.e+-]+([ ]+[-0-9.e+-]+)+$', clean_line):
                # Remove any row name
                values = re.split(r'\s+', clean_line)
                # If we have more values than headers, first might be row name
                if header and len(values) > len(header):
                    values = values[-(len(header)):]
                data_lines.append(values)
            else:
                # End of data section
                in_data_section = False
        
        # Start of data section could be marked by "> data"
        if line.startswith('> data') or line.startswith('> df'):
            in_data_section = True
            
    # If we found a header and data, create a dataframe
    if header and data_lines:
        try:
            # Convert strings to numeric values where possible
            numeric_data = []
            for row in data_lines:
                numeric_row = []
                for val in row:
                    try:
                        numeric_row.append(float(val))
                    except ValueError:
                        numeric_row.append(val)
                numeric_data.append(numeric_row)
                
            df = pd.DataFrame(numeric_data, columns=header)
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
    
    # If regular dataframe extraction failed, try simpler approach
    # Look for variables in output (e.g., "x1 = 1.23 2.34 3.45")
    variables = {}
    for line in lines:
        # Check for variable assignment
        match = re.match(r'>\s*([A-Za-z0-9_.]+)\s*<-\s*c\((.*)\)', line)
        if match:
            var_name = match.group(1)
            values_str = match.group(2)
            try:
                # Parse the values
                values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
                variables[var_name] = np.array(values)
            except ValueError:
                # If not numeric, might be strings
                values = [v.strip().strip('"\'') for v in values_str.split(',') if v.strip()]
                variables[var_name] = np.array(values)
    
    return variables

def synthetic_data_to_numpy(synthetic_data, target_var=None):
    """Convert synthetic data to numpy arrays.
    
    Args:
        synthetic_data: Dictionary with synthetic data information
        target_var: Name of target variable (optional)
        
    Returns:
        NumPy array(s)
    """
    # Get text output from synthetic data
    text_output = synthetic_data.get("results", {}).get("text_output", "")
    
    # Extract data
    data = extract_data_from_r_output(text_output)
    
    # If we got a DataFrame
    if isinstance(data, pd.DataFrame):
        # If target_var specified, separate features and target
        if target_var:
            if target_var in data.columns:
                X = data.drop(columns=[target_var]).values
                y = data[target_var].values
                return X, y
            else:
                # Try to guess - last column often the target
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                return X, y
        return data.values
    
    # If we got a dictionary of arrays
    elif isinstance(data, dict):
        if target_var and target_var in data:
            # Find feature variables (assume they start with x)
            feature_vars = [v for v in data.keys() if v.startswith('x')]
            if feature_vars:
                X = np.column_stack([data[v] for v in sorted(feature_vars)])
                y = data[target_var]
                return X, y
            
        # Return the entire dict if we can't process further
        return data
    
    # Default case - create some random data for demonstration
    print("Warning: Could not extract proper data, using random data for demonstration")
    X = np.random.normal(0, 1, (100, 3))
    if target_var:
        y = np.random.normal(0, 1, 100)
        return X, y
    return X

def synthetic_data_to_dataframe(synthetic_data):
    """Convert synthetic data to pandas DataFrame.
    
    Args:
        synthetic_data: Dictionary with synthetic data information
        
    Returns:
        pandas DataFrame
    """
    # Get text output from synthetic data
    text_output = synthetic_data.get("results", {}).get("text_output", "")
    
    # Extract data
    data = extract_data_from_r_output(text_output)
    
    # If we already got a DataFrame
    if isinstance(data, pd.DataFrame):
        return data
    
    # If we got a dictionary of arrays
    elif isinstance(data, dict):
        # Try to create a DataFrame from the dictionary
        try:
            # Find all arrays of the same length
            lengths = {k: len(v) for k, v in data.items() if hasattr(v, '__len__')}
            if lengths:
                most_common_len = max(lengths.values(), key=list(lengths.values()).count)
                valid_vars = {k: v for k, v in data.items() 
                             if hasattr(v, '__len__') and len(v) == most_common_len}
                return pd.DataFrame(valid_vars)
        except Exception as e:
            print(f"Error creating DataFrame from dict: {e}")
    
    # Default case - create some random data for demonstration
    print("Warning: Could not extract proper data, using random data for demonstration")
    X = np.random.normal(0, 1, (100, 3))
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['group'] = np.random.choice(['A', 'B', 'C'], size=100)
    df['value'] = 2 + 0.5*df['x1'] + 0.7*df['x2'] + np.random.normal(0, 1, 100)
    return df

def get_feature_names(synthetic_data):
    """Extract feature names from synthetic data.
    
    Args:
        synthetic_data: Dictionary with synthetic data information
        
    Returns:
        List of feature names
    """
    # Get text output from synthetic data
    text_output = synthetic_data.get("results", {}).get("text_output", "")
    
    # Extract data
    data = extract_data_from_r_output(text_output)
    
    # If we got a DataFrame
    if isinstance(data, pd.DataFrame):
        # Exclude obvious target columns
        exclude_cols = ['y', 'class', 'target', 'group', 'outcome', 'response', 'dependent']
        return [col for col in data.columns if col.lower() not in exclude_cols]
    
    # If we got a dictionary of arrays
    elif isinstance(data, dict):
        # Find feature variables (assume they start with x)
        feature_vars = [v for v in data.keys() if v.startswith('x')]
        if feature_vars:
            return sorted(feature_vars)
    
    # Default case
    return None 