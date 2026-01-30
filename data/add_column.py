import pandas as pd

def add_activity_class(df_ref: pd.DataFrame,
                       df_target: pd.DataFrame,
                       ref_molid_col: str = "mol_id",
                       ref_activity_col: str = "activity_class",
                       target_molid_col: str = "mol_id",
                       strict_unique_ref: bool = True,
                       coerce_molid_to_str: bool = True) -> pd.DataFrame:
    """
    Add activity_class from df_ref to df_target by matching mol_id.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference dataframe containing mol_id and activity_class.
    df_target : pd.DataFrame
        Target dataframe to which activity_class will be added.
    ref_molid_col : str
        Column name for mol_id in df_ref.
    ref_activity_col : str
        Column name for activity_class in df_ref.
    target_molid_col : str
        Column name for mol_id in df_target.
    strict_unique_ref : bool
        If True, raise error if df_ref has duplicate mol_id (ambiguous mapping).
        If False, duplicates are resolved by keeping the first occurrence.
    coerce_molid_to_str : bool
        If True, cast mol_id columns in both dfs to string to avoid dtype mismatches.

    Returns
    -------
    pd.DataFrame
        New dataframe = df_target + activity_class column.
    """
    # Defensive copies (avoid mutating caller's dfs)
    ref = df_ref[[ref_molid_col, ref_activity_col]].copy()
    out = df_target.copy()

    # Optional: align dtypes to avoid silent non-matches (int vs str)
    if coerce_molid_to_str:
        ref[ref_molid_col] = ref[ref_molid_col].astype(str)
        out[target_molid_col] = out[target_molid_col].astype(str)

    # Handle duplicate mol_id in reference
    if strict_unique_ref:
        dup_mask = ref[ref_molid_col].duplicated(keep=False)
        if dup_mask.any():
            dup_vals = ref.loc[dup_mask, ref_molid_col].unique()[:20]
            raise ValueError(
                f"Reference df has duplicate {ref_molid_col} values (showing up to 20): {dup_vals}. "
                f"Set strict_unique_ref=False to keep the first occurrence."
            )
    else:
        ref = ref.drop_duplicates(subset=[ref_molid_col], keep="first")

    # Map (fast and clean)
    lookup = ref.set_index(ref_molid_col)[ref_activity_col]
    out[ref_activity_col] = out[target_molid_col].map(lookup)

    return out


# ---- Example usage ----
if __name__ == "__main__":
    df1 = pd.read_csv("ref.csv")       # must contain mol_id, activity_class
    df2 = pd.read_csv("target.csv")    # must contain mol_id
    new_df = add_activity_class(df1, df2)
    new_df.to_csv('chembl_stereo_class.csv')
    # print(new_df.head())
    
