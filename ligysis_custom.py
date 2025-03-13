### IMPORTS ###

import os
import Bio
import sys
import math
import scipy
import pickle
import shutil
import logging
import argparse
import requests
import Bio.SeqIO
import prointvar
import Bio.AlignIO
import numpy as np
import configparser
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqUtils
from Bio import pairwise2
from scipy import cluster
import varalign.alignments
import scipy.stats as stats
import varalign.align_variants
from prointvar.pdbx import PDBXreader
from prointvar.pdbx import PDBXwriter
from scipy.spatial.distance import squareform
from prointvar.dssp import DSSPrunner, DSSPreader


## UTILITIES

def dump_pickle(data, pickle_out):
    """
    Dumps pickle.
    """
    with open(pickle_out, "wb") as f:
        pickle.dump(data, f)

def load_pickle(pickle_in):
    """
    Loads pickle.
    """
    with open(pickle_in, "rb") as f:
        data = pickle.load(f)
    return data

### DICTIONARIES AND LISTS

pdb_clean_suffixes = ["break_residues", "breaks"]

# simple_ions = [
#     "ZN", "MN", "CL", "MG", "CD", "NI", "NA", "IOD", "CA", "BR", "XE"
# ]

# acidic_ions = [
#     "PO4", "ACT", "SO4", "MLI", "CIT", "ACY", "VO4"
# ]

# non_relevant_ligs_manual = [
#     "DMS", "EDO", "HOH", "TRS", "GOL", "OGA", "FMN", "PG4", "PGR",
#     "MPD", "TPP", "MES", "PLP", "HYP", "CSO", "UNX", "EPE", "PEG",
#     "PGE", "DOD", "SUI"
# ]

# non_relevant = non_relevant_ligs_manual + simple_ions + acidic_ions

pdb_resnames = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",       ### ONE OF THESE MIGHT BE ENOUGH ###
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
]

aas_1l= [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "-"
]

aa_code = {
    "ALA" : 'A', "CYS" : 'C', "ASP" : 'D', "GLU" : 'E',
    "PHE" : 'F', "GLY" : 'G', "HIS" : 'H', "ILE" : 'I',
    "LYS" : 'K', "LEU" : 'L', "MET" : 'M', "ASN" : 'N',
    "PRO" : 'P', "GLN" : 'Q', "ARG" : 'R', "SER" : 'S',
    "THR" : 'T', "VAL" : 'V', "TRP" : 'W', "TYR" : 'Y',
    "PYL" : 'O', "SEC" : 'U', "HYP" : 'P', "CSO" : 'C', # WEIRD ONES
    "SUI" : 'D',
}

cif_cols_order = [
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
    # "label_entity_id",
    "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
    "B_iso_or_equiv", "pdbx_formal_charge", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"
]

chimeraX_commands = [
    "color white; set bgColor white",
    "set silhouette ON; set silhouetteWidth 2; set silhouetteColor black",
    "~disp; select ~protein; ~select : HOH; ~select ::binding_site==-1; disp sel; ~sel",
    "surf; surface color white; transparency 70 s;"
]

consvar_class_colours = [
    "royalblue", "green", "grey", "firebrick", "orange"
]

interaction_to_color = { # following Arpeggio's colour scheme
    'clash': '#000000',
    'covalent':'#999999',
    'vdw_clash': '#999999',
    'vdw': '#999999',
    'proximal': '#999999',
    'hbond': '#f04646',
    'weak_hbond': '#fc7600',
    'xbond': '#3977db', #halogen bond
    'ionic': '#e3e159',
    'metal_complex': '#800080',
    'aromatic': '#00ccff',
    'hydrophobic': '#006633',
    'carbonyl': '#ff007f',
    'polar': '#f04646',
    'weak_polar': '#fc7600',
}

bss_colors = load_pickle("./OTHER/colors.pkl") # sample colors

headings = ["ID", "RSA", "DS", "MES", "Size", "Cluster", "FS"]

wd = os.getcwd()

### CONFIG FILE READING AND VARIABLE SAVING

config = configparser.ConfigParser()
config.read("ligysis_config.txt")

stamp_bin = config["paths"].get("stamp_bin")
transform_bin = config["paths"].get("transform_bin")
clean_pdb_python_bin = config["paths"].get("clean_pdb_python_bin")
clean_pdb_bin = config["paths"].get("clean_pdb_bin")
arpeggio_python_bin = config["paths"].get("arpeggio_python_bin")
arpeggio_bin = config["paths"].get("arpeggio_bin")

gnomad_vcf = config["paths"].get("gnomad_vcf")
swissprot_path = config["paths"].get("swissprot")
ensembl_sqlite_path = config["paths"].get("ensembl_sqlite")

stampdir = config["paths"].get("stampdir")

### FUNCTIONS

## UTILITIES FUNCTIONS

def add_double_quotes_for_single_quote(df, columns = ["auth_atom_id", "label_atom_id"]):
    """
    Adds double quotes to values with single quotes. This is needed for atoms with single quotes in their names.
    """
    for column in columns:
        df[column] = df[column].apply(
            lambda x: x if str(x).startswith('"') and str(x).endswith('"') else f'"{x}"' if "'" in str(x) else x
        )
    return df

## SETUP FUNCTIONS

def setup_dirs(dirs):
    """
    Creates directories if they do not exist.
    """
    for dirr in dirs:
        if os.path.isdir(dirr):
            continue
        else:
            os.mkdir(dirr)

## UNIPROT NAMES FUNCTION

def get_uniprot_info(uniprot_id):
    """
    Fetches UniProt ID, UniProt entry, and protein name for the given UniProt ID.
    
    Parameters:
    uniprot_id (str): UniProt ID to fetch the information for.
    
    Returns:
    dict: A dictionary with UniProt ID, UniProt entry, and protein name.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        uniprot_id = data.get('primaryAccession', 'N/A')
        uniprot_entry = data.get('uniProtkbId', 'N/A')
        protein_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
        
        return {
            'up_id': uniprot_id,
            'up_entry': uniprot_entry,
            'prot_name': protein_name
        }
    except:
        return {"up_id": uniprot_id, "up_entry": "", "prot_name": ""}

## STAMPING FUNCTIONS

def generate_STAMP_domains(wd, pdbs_dir, domains_out, roi = "ALL"):
    """
    Genereates domains file, needed to run STAMP.
    """
    pdb_files = [file for file in os.listdir(pdbs_dir) if file.endswith(".pdb")]
    if pdb_files == []:
        pdb_files = [file for file in os.listdir(pdbs_dir) if file.endswith(".ent")] # trying .ent

    rel_pdbs_dir = os.path.relpath(pdbs_dir, wd)
    
    with open(domains_out, "w+") as fh:
        for pdb in pdb_files: # THIS SHOULD ALWAYS BE EITHER .PDB OR .ENT
            pdb_root, pdb_ext = os.path.splitext(pdb)
            pdb_name = pdb_root.replace(".clean", "") # can't rely on the split, as the pdb might have a . in the name
            fh.write("{} {} {{{}}}\n".format(os.path.join(rel_pdbs_dir, pdb), pdb_name + ".supp", roi))

def stamp(domains, prefix, out):
    """
    Runs STAMP using the domains file as input.
    """
    if "STAMPDIR" not in os.environ:
        os.environ["STAMPDIR"] = stampdir
    args = [
        stamp_bin, "-l", domains, "-rough", "-n", ### STAMP STILL CRASHES IF PATHS ARE TOO LONG ###
        str(2), "-prefix", prefix, ">", out
    ]
    cmd = " ".join(args)
    exit_code = os.system(cmd)
    return cmd, exit_code

def transform(matrix):
    """
    Runs TRANSFORM to obtain set of transformed coordinates.
    """
    if "STAMPDIR" not in os.environ:
        os.environ["STAMPDIR"] = stampdir
    args = [transform_bin, "-f", matrix, "-het"]
    cmd = " ".join(args)
    exit_code = os.system(cmd)
    return cmd, exit_code

def fnames_from_domains(domains_out):
    """
    Returns a list of the pdb file names from the STAMP domains file.
    """
    with open(domains_out) as f:
        lines = f.readlines()
    fnames = []
    for line in lines:
        fnames.append(line.split()[1] + ".pdb")
    return fnames

def move_supp_files(struc_files, supp_pdbs_dir, cwd):
    """
    Moves set of supperimposed coordinate files to appropriate directory.
    """
    for file in struc_files:
        if os.path.isfile(os.path.join(cwd, file)):
            shutil.move(
                os.path.join(cwd, file),
                os.path.join(supp_pdbs_dir, file)
            )

def move_stamp_output(prefix, stamp_out_dir, cwd):
    """
    Moves STAMP output files to appropriate directory.
    """
    stamp_files = sorted([file for file in os.listdir(cwd) if prefix in file]) + ["stamp_rough.trans"]
    for file in stamp_files:
        filepath = os.path.join(cwd, file)
        if os.path.isfile(filepath):
            shutil.move(filepath, os.path.join(stamp_out_dir, file))
    out_from = os.path.join(cwd, prefix + ".out")
    out_to = os.path.join(stamp_out_dir, prefix + ".out")
    doms_from = os.path.join(cwd, prefix + ".domains")
    doms_to = os.path.join(stamp_out_dir, prefix + ".domains")
    if os.path.isfile(out_from):
        shutil.move(out_from, out_to)
    if os.path.isfile(doms_from):
        shutil.move(doms_from, doms_to)

def simplify_pdb(supp_file, simple_file, struc_fmt = "mmcif"):
    """
    Simplifies pdb file by removing all non-ATOM records.
    """
    df = PDBXreader(inputfile = supp_file).atoms(format_type = struc_fmt, excluded=())
    df = add_double_quotes_for_single_quote(df)
    df_hetatm = df.query('group_PDB != "ATOM"')
    if df_hetatm.empty:
        return # no HETATM records, no simple file is written
    else:
        w = PDBXwriter(outputfile = simple_file)
        w.run(df_hetatm, format_type = struc_fmt, category = "auth")

## LIGAND FUNCTIONS

def get_lig_data(cifs_dir, ligs_df_path, struc_fmt = "mmcif"):
    """
    From a directory containing a set of structurally superimposed pdbs,
    writes a .pkl file indicating the name, chain and residue number of the
    ligand(s) of interest in every cif.
    """
    ligs_df = pd.DataFrame([])
    simple_cif_files = [file for file in os.listdir(cifs_dir) if file.endswith(".cif")]
    for struc in simple_cif_files:
        struc_path = os.path.join(cifs_dir, struc)
        df = PDBXreader(inputfile = struc_path).atoms(format_type = struc_fmt, excluded=())
        hetatm_df = df.query('group_PDB == "HETATM"')
        ligs = hetatm_df.label_comp_id.unique().tolist()
        #lois = [lig for lig in ligs if lig not in non_relevant]
        lois = ligs #currently taking all ligands
        for loi in lois:
            loi_df = hetatm_df.query('label_comp_id == @loi')
            #lois_df_un = loi_df.drop_duplicates(["label_comp_id", "label_asym_id"])[["label_comp_id", "label_asym_id", "auth_seq_id"]]
            lois_df_un = loi_df.drop_duplicates(["label_comp_id", "auth_asym_id", "auth_seq_id"])[["label_comp_id", "auth_asym_id", "auth_seq_id"]] # changing this to auth, cause it seems pdbe-arpeggio uses auth (except for _com_id).
            lois_df_un["struc_name"] = struc
            ligs_df = ligs_df.append(lois_df_un)
    ligs_df = ligs_df[["struc_name", "label_comp_id", "auth_asym_id", "auth_seq_id"]]
    ligs_df.to_pickle(ligs_df_path)
    return ligs_df

## SIFTS FUNCTIONS

def get_protein_sequence(uniprot_id):
    """
    Retrieves the protein sequence for a given UniProt ID.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.ok:
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:]) # Removing the description line (first line starting with ">")
        return sequence
    else:
        raise ValueError(f"Error fetching data for UniProt ID {uniprot_id}")

def retrieve_mapping_from_struc(struc, uniprot_id, struc_dir, mappings_dir, struc_fmt = "mmcif"):
    """
    Retrieves the mapping between the UniProt sequence and the PDB sequence by doing an alignment.
    """
    input_struct = os.path.join(struc_dir, struc)
    pdb_structure = PDBXreader(inputfile = input_struct).atoms(format_type = struc_fmt, excluded=()) # ProIntVar reads the local file

    sequence = get_protein_sequence(uniprot_id)

    pps = pdb_structure.query('group_PDB == "ATOM"')[['label_comp_id', 'auth_asym_id', 'auth_seq_id']].drop_duplicates().groupby('auth_asym_id')  # groupby chain
    pdb_chain_seqs = [(chain, SeqUtils.seq1(''.join(seq['label_comp_id'].values)), seq['auth_seq_id'].values) for chain, seq in pps] # list of tuples like: [(chain_id, chain_seq, [chain resnums])]
    alignments = [pairwise2.align.globalxs(sequence, chain_seq[1], -5, -1) for chain_seq in pdb_chain_seqs] # list of lists of tuples containing SwissProt seq - PDB chain seq pairwise alignment
    
    maps = []
    for pdb_chain_seq, alignment in zip(pdb_chain_seqs, alignments):
        PDB_UniProt_map = pd.DataFrame(
            [(i, x) for i, x in enumerate(alignment[0][1], start=1)],  # create aligned PDB sequences to dataframe
            columns=['UniProt_ResNum', 'PDB_ResName']
        )
        PDB_UniProt_map = PDB_UniProt_map.assign(UniProt_ResName = list(alignment[0][0]))
        PDB_index = PDB_UniProt_map.query('PDB_ResName != "-"').index
        PDB_UniProt_map = PDB_UniProt_map.assign(PDB_ResNum = pd.Series(pdb_chain_seq[2], index = PDB_index)) # adds PDB_ResNum column
        PDB_UniProt_map = PDB_UniProt_map.assign(PDB_ChainID = pd.Series(pdb_chain_seq[0], index = PDB_index)) # adds PDB_ChainId column
        maps.append(PDB_UniProt_map)
    prointvar_mapping = pd.concat(maps)
    prointvar_mapping = prointvar_mapping[['UniProt_ResNum','UniProt_ResName','PDB_ResName','PDB_ResNum','PDB_ChainID']]
    prointvar_mapping = prointvar_mapping[~prointvar_mapping.PDB_ResNum.isnull()]
    struc_root, _ = os.path.splitext(struc)
    struc_name = struc_root.replace(".supp", "") # can't rely on the split, as the pdb might have a . in the name
    prointvar_mapping_csv = os.path.join(mappings_dir, struc_name + "_mapping.csv")
    prointvar_mapping.PDB_ResNum = prointvar_mapping.PDB_ResNum.astype(str) # PDB_ResNum is a string, not an integer
    prointvar_mapping.to_csv(prointvar_mapping_csv, index = False)
    return prointvar_mapping

def get_pseudo_mapping_from_struc(struc, struc_dir, mappings_dir, struc_fmt = "mmcif"):
    """
    Retrieves a pseudo-mapping for each structure when a UniProt ID is not provided.
    """
    input_struct = os.path.join(struc_dir, struc)
    pdb_structure = PDBXreader(inputfile = input_struct).atoms(format_type = struc_fmt, excluded=()) # ProIntVar reads the local file

    pps = pdb_structure.query('group_PDB == "ATOM"')[['label_comp_id', 'auth_asym_id', 'auth_seq_id']].drop_duplicates().groupby('auth_asym_id')  # groupby chain
    pdb_chain_seqs = [(chain, SeqUtils.seq1(''.join(seq['label_comp_id'].values)), seq['auth_seq_id'].values) for chain, seq in pps] # list of tuples like: [(chain_id, chain_seq, [chain resnums])]
    
    data = {
        "UniProt_ResNum": [],
        "UniProt_ResName": [],
        "PDB_ResName": [],
        "PDB_ResNum": [],
        "PDB_ChainID": [],
    }

    for chain in pdb_chain_seqs:
        data["UniProt_ResNum"].extend([int(el) for el in chain[2]]) # not changing column names, but not actually UniProt
        data["UniProt_ResName"].extend(list(chain[1])) # not changing column names, but not actually UniProt
        data["PDB_ResName"].extend(list(chain[1]))
        data["PDB_ResNum"].extend(chain[2])
        data["PDB_ChainID"].extend(chain[0]*len(chain[2]))

    pseudo_mapping = pd.DataFrame(data)
    pseudo_mapping = pseudo_mapping[~pseudo_mapping.PDB_ResNum.isnull()]
    struc_root, _ = os.path.splitext(struc)
    struc_name = struc_root.replace(".supp", "") # can't rely on the split, as the pdb might have a . in the name
    pseudo_mapping_csv = os.path.join(mappings_dir, struc_name + "_mapping.csv")
    pseudo_mapping.to_csv(pseudo_mapping_csv, index = False)
    return pseudo_mapping

## DSSP FUNCTIONS

def run_dssp(struc, supp_pdbs_dir, dssp_dir):
    """
    Runs DSSP, saves and return resulting output dataframe
    """
    struc_root, _ = os.path.splitext(struc)
    dssp_csv = os.path.join(dssp_dir, struc_root + ".csv") # output csv filepath
    dssp_out = os.path.join(dssp_dir, struc_root + ".dssp")
    struc_in = os.path.join(supp_pdbs_dir, struc)
    DSSPrunner(inputfile = struc_in, outputfile = dssp_out).write()            # runs DSSP
    dssp_data = DSSPreader(inputfile = dssp_out).read()            # reads DSSP output
    dssp_data = dssp_data.rename(index = str, columns = {"RES": "PDB_ResNum"})
    dssp_data.PDB_ResNum = dssp_data.PDB_ResNum.astype(str)
    dssp_cols = ["PDB_ResNum", "SS", "ACC", "KAPPA", "ALPHA", "PHI", "PSI", "RSA"]    # selects subset of columns
    dssp_data.to_csv(dssp_csv, index = False)
    return dssp_data[dssp_cols]

## ARPEGGIO FUNCTIONS

def run_clean_pdb(pdb_path):
    """
    Runs pdb_clean.py to prepare files for Arpeggio.
    """
    args = [
        clean_pdb_python_bin, clean_pdb_bin, pdb_path
    ]
    cmd = " ".join(args)
    exit_code = os.system(cmd)
    return cmd, exit_code

def run_arpeggio(pdb_path, lig_sel, out_dir):
    """
    runs Arpeggio
    """
    args = [
        arpeggio_python_bin, arpeggio_bin, pdb_path,
        "-s", lig_sel, "-o", out_dir, "--mute"
    ]
    cmd = " ".join(args)
    exit_code = os.system(cmd)
    return exit_code, cmd

def switch_columns(df, names):
    """
    Switches columns in Arpeggio DataFrame, so that ligand atoms and protein
    atoms are always on the same column.
    """

    columns_to_switch = [
        'auth_asym_id', 'auth_atom_id',
        'auth_seq_id', 'label_comp_id'
    ]

    for index, row in df.iterrows(): # Iterate through the DataFrame and switch columns where necessary
        if row['label_comp_id_end'] in names:
            for col in columns_to_switch:
                bgn_col = f"{col}_bgn"
                end_col = f"{col}_end"
                df.at[index, bgn_col], df.at[index, end_col] = df.at[index, end_col], df.at[index, bgn_col]

    return df

def map_values(row, pdb2up):
    """
    maps UniProt ResNums from SIFTS dictionary from CIF file to Arpeggio dataframe.
    """
    try:
        return pdb2up[row.auth_asym_id_end][row.auth_seq_id_end]
    except KeyError:
        log.debug(f'Residue {row.auth_seq_id_end} chain {row.auth_asym_id_end} has no mapping to UniProt')
        return np.nan # if there is no mapping, return NaN

def create_resnum_mapping_dicts(df):
    """
    Creates a dictionary with the mapping between PDB_ResNum and UniProt_ResNum
    froma  previously created dataframe.
    """
    pdb2up = {}
    up2pdb = {}
    
    for index, row in df.iterrows():
        chain_id = row['PDB_ChainID']
        pdb_resnum = row['PDB_ResNum']
        uniprot_resnum = row['UniProt_ResNum']
        
        if chain_id not in pdb2up: # Initialise dictionary for the chain ID if it doesn't exist
            pdb2up[chain_id] = {}
        if uniprot_resnum not in up2pdb:
            up2pdb[uniprot_resnum] = []
        
        pdb2up[chain_id][str(pdb_resnum)] = uniprot_resnum 
        up2pdb[uniprot_resnum].append((chain_id, str(pdb_resnum)))
    
    return pdb2up, up2pdb

def process_arpeggio_df(arp_df, pdb_id, ligand_names, pdb2up):
    """
    Process Arpeggio Df to put in appropriate
    format to extract fingerprings. Also filter out
    non-relevant interactions.
    """
    
    arp_df_end_expanded = arp_df['end'].apply(pd.Series)
    arp_df_bgn_expanded = arp_df['bgn'].apply(pd.Series)

    arp_df = arp_df.join(arp_df_end_expanded).drop(labels='end', axis=1)
    arp_df = arp_df.join(arp_df_bgn_expanded, lsuffix = "_end", rsuffix = "_bgn").drop(labels='bgn', axis = 1)

    arp_df.auth_seq_id_bgn = arp_df.auth_seq_id_bgn.astype(str)
    arp_df.auth_seq_id_end = arp_df.auth_seq_id_end.astype(str)

    inter_df = arp_df.query('interacting_entities == "INTER" & type == "atom-atom"').copy().reset_index(drop = True)
    inter_df = inter_df[inter_df['contact'].apply(lambda x: 'clash' not in x)].copy().reset_index(drop = True) # filtering out clashes
    inter_df = inter_df.query('label_comp_id_bgn in @pdb_resnames or label_comp_id_end in @pdb_resnames').copy().reset_index(drop = True) # filtering out ligand-ligand interactions
    
    if inter_df.empty:
        log.warning("No protein-ligand interaction  for {}".format(pdb_id))
        return inter_df, "no-PL-inters"
    
    inter_df = inter_df.query('label_comp_id_bgn in @ligand_names or label_comp_id_end in @ligand_names').copy().reset_index(drop = True) # filtering out non-LOI interactions (only to avoid re-running Arpeggio, once it has been run with wrong selection)
    
    switched_df = switch_columns(inter_df, ligand_names)
    switched_df = switched_df.query('label_comp_id_end in @pdb_resnames').copy() # filtering out non-protein-ligand interactions
    switched_df["UniProt_ResNum_end"] = switched_df.apply(lambda row: map_values(row, pdb2up), axis=1) # Apply the function and create a new column
    switched_df = switched_df.sort_values(by=["auth_asym_id_end", "UniProt_ResNum_end", "auth_atom_id_end"]).reset_index(drop = True)
    
    return switched_df, "OK"

def get_inters(fingerprints_dict):
    """
    Returns all ligand fingerprints from fingerprints dict.
    """
    return [v for v in fingerprints_dict.values()]

def get_labs(fingerprints_dict):
    """
    Returns all ligand labels from fingerprints dict.
    """
    return [k for k in fingerprints_dict.keys()]

## mmcif dict with ProIntVar

def generate_dictionary(mmcif_file):
    """
    Generates a dictionary of coordinates from an mmcif file.
    """
    cif_df = PDBXreader(inputfile = mmcif_file).atoms(format_type = "mmcif", excluded=())
    
    keys = list(zip(cif_df['auth_asym_id'], cif_df['label_comp_id'], cif_df['auth_seq_id'], cif_df['auth_atom_id']))  # Create the keys using vectorised operations
    
    values = cif_df[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().tolist() # Create the values as a list of lists (x, y, z)

    result = dict(zip(keys, values)) # Combine the keys and values into a dictionary

    return result

def determine_width(interactions):
    """
    Generates cylinder width for 3DMol.js interaction
    representation depending on Arpeggio contact
    fingerprint.
    """
    return 0.125 if 'vdw_clash' in interactions else 0.0625

def determine_color(interactions):
    """
    Generates cylinder colour for 3DMol.js interaction
    representation depending on Arpeggio contact
    fingerprint.
    """
    undef = ['covalent', 'vdw', 'vdw_clash', 'proximal']
    if len(interactions) == 1 and interactions[0] in undef:
        return '#999999'
    else:
        colors = [interaction_to_color[interaction] for interaction in interactions if interaction in interaction_to_color and interaction not in undef]
        if colors:
            return colors[0]
        else:
            log.critical("No color found for {}".format(interactions))
            return None  # Return the first color found, or None if no match

### FUNCTIONS FOR SITE DEFINITION

def get_intersect_rel_matrix(binding_ress):
    """
    Given a set of ligand binding residues, calcualtes a
    similarity matrix between all the different sets of ligand
    binding residues.
    """
    inters = {i: {} for i in range(len(binding_ress))}
    for i in range(len(binding_ress)):
        inters[i][i] = intersection_rel(binding_ress[i], binding_ress[i])
        for j in range(i+1, len(binding_ress)):
            inters[i][j] = intersection_rel(binding_ress[i], binding_ress[j])
            inters[j][i] = inters[i][j]
    return inters

def intersection_rel(l1, l2):
    """
    Calculates relative intersection.
    """
    len1 = len(l1)
    len2 = len(l2)
    I_max = min([len1, len2])
    I = len(list(set(l1).intersection(l2)))
    return I/I_max

def write_chimeraX_attr(cluster_id_dict, trans_dir, attr_out): # cluster_id_dict is now the new one with orig_label_asym_id
    """
    Gets chimeraX atom specs, binding site ids, and paths
    to pdb files to generate the attribute files later, and
    eventually colour models. 
    """
    trans_files = [f for f in os.listdir(trans_dir) if f.endswith(".cif")]
    order_dict = {k : i+1 for i, k in enumerate(trans_files)}
    
    defattr_lines = []

    for k, v in cluster_id_dict.items():

        ld = k.split("_") # stands for lig data

        struc_id, lig_resname, lig_chain_id, lig_resnum  = ["_".join(ld[:-3]), ld[-3], ld[-2], ld[-1]] # this way, indexing from the end should cope with any struc_ids

        if k in cluster_id_dict:
            defattr_line = "\t#{}/{}:{}\t{}\n\n".format(order_dict[f'{struc_id}.simp.cif'], lig_chain_id, lig_resnum, v)
        else:
            defattr_line = "\t#{}/{}:{}\t{}\n\n".format(order_dict[f'{struc_id}.simp.cif'], lig_chain_id, lig_resnum, "-1")
        defattr_lines.append(defattr_line)
        
    with open(attr_out, "w") as out:
        out.write("attribute: binding_site\n\n")
        out.write("match mode: 1-to-1\n\n")
        out.write("recipient: residues\n\n")
        for i in sorted(defattr_lines):
            out.write(i)
    return 

def write_chimeraX_script(chimera_script_out, trans_dir, attr_out, chX_session_out, chimeraX_commands, cluster_ids):
    """
    Writes a chimeraX script to colour and format.
    """
    trans_files = [f for f in os.listdir(trans_dir) if f.endswith(".cif")] ### FIXME format
    with open(chimera_script_out, "w") as out:
        out.write("# opening files\n\n")
        for f in trans_files:
            out.write("open {}\n\n".format(f))
        out.write("# opening attribute file\n\n")
        out.write("open {}\n\n".format(attr_out))
        out.write("# colouring and formatting for visualisation\n\n")
        for cmxcmd in chimeraX_commands:
            out.write("{}\n\n".format(cmxcmd))
        for cluster_id in cluster_ids:
            out.write(f'col ::binding_site=={cluster_id} {bss_colors[cluster_id]};\n')
        out.write("save {}\n\n".format(chX_session_out))
    return

def get_residue_bs_membership(cluster_ress):
    """
    Returns a dictionary indicating to which ligand binding
    site each ligand binding residue is found in. A residue
    might contribute to more than one adjacent binding site.
    """
    all_bs_ress = []
    for v in cluster_ress.values():
        all_bs_ress.extend(v)
    all_bs_ress = sorted(list(set(all_bs_ress)))
    
    bs_ress_membership_dict = {}
    for bs_res in all_bs_ress:
        bs_ress_membership_dict[bs_res] = []
        for k, v in cluster_ress.items():
            if bs_res in v:
                bs_ress_membership_dict[bs_res].append(k) # which binding site each residue belongs to
    return bs_ress_membership_dict

def get_cluster_membership(cluster_id_dict):
    """
    Creates a dictionary indicating to which cluster
    each ligand binds to.
    """
    membership_dict = {}
    for k, v in cluster_id_dict.items():
        if v not in membership_dict:
            membership_dict[v] = []
        membership_dict[v].append(k)
    return membership_dict

def get_all_cluster_ress(membership_dict, fingerprints_dict):
    """
    Given a membership dict and a fingerprint dictionary,
    returns a dictionary that indicates the protein residues
    forming each binding site.
    """
    binding_site_res_dict = {}
    for k, v in membership_dict.items():
        if k not in binding_site_res_dict:
            binding_site_res_dict[k] = []
        for v1 in v:
            binding_site_res_dict[k].extend(fingerprints_dict[v1])
    binding_site_res_dict = {k: sorted(list(set(v))) for k, v in binding_site_res_dict.items()}
    return binding_site_res_dict

### CONSERVATION + VARIATION FUNCTIONS

def create_alignment_from_struc(example_struc, fasta_path, struc_fmt = "mmcif", n_it = 3, seqdb = swissprot_path):
    """
    Given an example structure, creates and reformats an MSA.
    """
    main_chain_seq = get_seq_from_pdb(example_struc, struc_fmt = struc_fmt)
    create_fasta_from_seq(main_chain_seq, fasta_path) # CREATES FASTA FILE FROM PDB FILE
    fasta_root, _ = os.path.splitext(fasta_path)    
    hits_out = "{}.out".format(fasta_root)
    hits_aln = "{}.sto".format(fasta_root)
    hits_aln_rf = "{}_rf.sto".format(fasta_root)    
    jackhmmer(fasta_path, hits_out, hits_aln , n_it = n_it, seqdb = seqdb,) # RUNS JACKHAMMER USING AS INPUT THE SEQUENCE FROM THE PDB AND GENERATES ALIGNMENT
    add_acc2msa(hits_aln, hits_aln_rf)

def get_seq_from_pdb(pdb_path, struc_fmt = "mmcif"): # SELECTS FIRST CHAIN. CURRENTLY ONLY WORKS WITH ONE CHAIN
    """
    Generates aa sequence string from a pdb coordinates file.
    """
    struc = PDBXreader(pdb_path).atoms(format_type = struc_fmt, excluded=())
    chains = sorted(list(struc.auth_asym_id.unique())) ### TODO: we generate sequence only for the first chain. this means this will only work for MONOMERIC PROTEINS
    main_chain = chains[0]
    main_chain_seq = "".join([aa_code[aa] for aa in struc.query('group_PDB == "ATOM" & auth_asym_id == @main_chain').drop_duplicates(["auth_seq_id"]).label_comp_id.tolist()])
    return main_chain_seq

def create_fasta_from_seq(seq, out):
    """
    Saves input sequence to fasta file to use as input for jackhmmer.
    """
    with open(out, "w+") as fh:
        fh.write(">query\n{}\n".format(seq))

def jackhmmer(seq, hits_out, hits_aln, n_it = 3, seqdb = swissprot_path):
    """
    Runs jackhmmer on an input seq for a number of iterations and returns exit code, should be 0 if all is OK.
    """
    args = ["jackhmmer", "-N", str(n_it), "-o", hits_out, "-A", hits_aln, seq, seqdb]
    cmd = " ".join(args)
    exit_code = os.system(cmd)
    return cmd, exit_code

def add_acc2msa(aln_in, aln_out, fmt_in = "stockholm"):
    """
    Modifies AC field of jackhmmer alignment in stockholm format.
    
    :param aln_in: path of input alignment
    :type aln_in: str, required
    :param aln_out: path of output alignment
    :type aln_in: str, required
    :param fmt_in: input and output MSA format
    :type aln_in: str, defaults to stockholm
    """
    aln = Bio.SeqIO.parse(aln_in, fmt_in)
    recs = []
    for rec in aln:
        if rec.id == "query":
            rec.annotations["accession"] = "QUERYSEQ" # ADDS ACCESSION FIELD TO QUERY SEQUENCE
            rec.annotations["start"] = 1
            rec.annotations["end"] = len(rec.seq)
        else:
            rec.annotations["accession"] = rec.id.split("|")[1]
        recs.append(rec)
    Bio.SeqIO.write(recs, aln_out, fmt_in)

def get_target_prot_cols(msa_in, msa_fmt = "stockholm"): 
    """
    Returns list of MSA col idx that are popualted on the protein target.
    """
    seqs = [str(rec.seq) for rec in Bio.SeqIO.parse(msa_in, msa_fmt) if "query" in rec.id]
    occupied_cols = [i+1 for seq in seqs for i, el in enumerate(seq) if el != "-"]
    return sorted(list(set(occupied_cols)))

def calculate_shenkin(aln_in, aln_fmt, out = None):
    """
    Given an MSA, calculates Shenkin ans occupancy, gap
    percentage for all columns.
    """
    cols = in_columns(aln_in, aln_fmt)
    scores = []
    occ = []
    gaps = []
    occ_pct = []
    gaps_pct = []
    for k, v in cols.items():
        scores.append(get_shenkin(k, v))
        stats = (get_stats(v))
        occ.append(stats[0])
        gaps.append(stats[1])
        occ_pct.append(stats[2])
        gaps_pct.append(stats[3])
    df = pd.DataFrame(list(zip(list(range(1,len(scores)+1)),scores, occ,gaps, occ_pct, gaps_pct)), columns = ["col", "shenkin", "occ", "gaps", "occ_pct", "gaps_pct"])
    if out != None:
        df.to_pickle(out)
    return df

def get_stats(col):
    """
    for a given MSA column, calculates some basic statistics
    such as column residue occupancy ang gaps
    """
    n_seqs = len(col)
    gaps = col.count("-")
    occ = n_seqs - gaps
    occ_pct = round(100*(occ/n_seqs), 2)
    gaps_pct = round(100-occ_pct, 2)
    return occ, gaps, occ_pct, gaps_pct

def in_columns(aln_in, infmt):
    """
    Returns dictionary in which column idx are the key
    and a list containing all aas aligned to that column
    is the value.
    """
    aln = Bio.AlignIO.read(aln_in, infmt)
    n_cols = len(aln[0])
    cols = {}
    for col in range(1,n_cols+1):
        cols[col] = []
    for row in aln:
        seq = str(row.seq)
        for i in range(0, len(seq)):
            cols[i+1].append(seq[i])
    return cols

def get_shenkin(i_col, col):
    """
    Calculates Shenkin score for an MSA column.
    """
    S = get_entropy(get_freqs(i_col, col))
    return round((2**S)*6,2)

def get_freqs(i_col, col):
    """
    Calculates amino acid frequences for a given MSA column.
    """

    abs_freqs = {aa: 0 for aa in aas_1l}
    non_standard_aas = {}
    for aa in col:
        aa = aa.upper()
        if col.count("-") == len(col):
            abs_freqs["-"] = 1
            return abs_freqs
        if aa in aas_1l:
            abs_freqs[aa] += 1
        else:
            if aa not in non_standard_aas:
                non_standard_aas[aa] = 0
            non_standard_aas[aa] += 1
    all_ns_aas = sum(non_standard_aas.values())
    if all_ns_aas != 0:
        log.warning("Column {} presents non-standard AAs: {}".format(str(i_col), non_standard_aas))
    rel_freqs = {k: v/(len(col) - all_ns_aas) for k, v in abs_freqs.items()}
    return rel_freqs

def get_entropy(freqs):
    """
    Calculates Shannon's entropy from a set of aa frequencies.
    """
    S = 0
    for f in freqs.values():
        if f != 0:
            S += f*math.log2(f)
    return -S

def format_shenkin(shenkin, prot_cols, out = None):
    """
    Formats conservation dataframe and also
    calculates two normalised versions of it.
    """
    shenkin_filt = shenkin[shenkin.col.isin(prot_cols)].copy()
    shenkin_filt.index = range(1, len(shenkin_filt) + 1) # CONTAINS SHENKIN SCORE, OCCUPANCY/GAP PROPORTION OF CONSENSUS COLUMNS
    min_shenkin = min(shenkin_filt.shenkin)
    max_shenkin = max(shenkin_filt.shenkin)
    shenkin_filt.loc[:, "rel_norm_shenkin"] = round(100*(shenkin_filt.shenkin - min_shenkin)/(max_shenkin - min_shenkin), 2) # ADDING NEW COLUMNS WITH DIFFERENT NORMALISED SCORES
    shenkin_filt.loc[:, "abs_norm_shenkin"] = round(100*(shenkin_filt.shenkin - 6)/(120 - 6), 2)
    if out != None:
        shenkin_filt.to_pickle(out)
    return shenkin_filt

def get_human_subset_msa(aln_in, human_msa_out, fmt_in = "stockholm"):
    """
    Creates a subset MSA containing only human sequences.
    """
    msa = Bio.AlignIO.read(aln_in, fmt_in)
    human_recs = []
    for rec in msa:
        if "HUMAN" in rec.name:
            human_recs.append(rec)
    Bio.SeqIO.write(human_recs, human_msa_out, fmt_in)

def cp_sqlite(wd, og_path = ensembl_sqlite_path):
    """
    Copies ensembl_cache.sqlite
    to execution directory.
    """
    hidden_var_dir = os.path.join(wd, ".varalign")
    sqlite_name = os.path.basename(og_path)
    if not os.path.isdir(hidden_var_dir):
        os.mkdir(hidden_var_dir)
    else:
        pass
    cp_path = os.path.join(hidden_var_dir, sqlite_name)
    shutil.copy(og_path, cp_path)
    return cp_path

def rm_sqlite(cp_path):
    """
    Removes ensembl_cache.sqlite
    from execution directory.
    """
    hidden_var_dir = os.path.dirname(cp_path)
    os.remove(cp_path)
    os.rmdir(hidden_var_dir)

def format_variant_table(df, col_mask, vep_mask = ["missense_variant"], tab_format = "gnomad"):
    """
    Formats variant table, by gettint rid of empty rows that are not human sequences,
    changning column names and only keeping those variants that are of interest and
    are present in columns of interest.
    """
    df_filt = df.copy(deep = True)
    df_filt.reset_index(inplace = True)
    if tab_format == "gnomad":
        df_filt.columns = [" ".join(col).strip() for col in df_filt.columns.tolist()]
    df_filt.columns = [col.lower().replace(" ", "_") for col in df_filt.columns.tolist()]
    df_filt = df_filt[df_filt.source_id.str.contains("HUMAN")]
    df_filt = df_filt.dropna(subset = ["vep_consequence"])
    df_filt = df_filt[df_filt.vep_consequence.isin(vep_mask)]
    df_filt = df_filt[df_filt.alignment_column.isin(col_mask)]
    return df_filt

def get_missense_df(aln_in, variants_df, shenkin_aln, prot_cols, aln_out, aln_fmt = "stockholm", get_or = True):
    """
    Generates a dataframe for the subset of human sequences with variants
    mapping to them. Calculates shenkin, and occupancy data, and then
    enrichment in variants.
    """
    variants_aln = generate_subset_aln(aln_in, aln_fmt, variants_df, aln_out)
    if variants_aln == "":
        return pd.DataFrame()
    variants_aln_info = calculate_shenkin(variants_aln, aln_fmt)
    variants_aln_info = variants_aln_info[variants_aln_info.col.isin(prot_cols)]
    vars_df = pd.DataFrame(variants_df.alignment_column.value_counts().reindex(prot_cols, fill_value = 0).sort_index()).reset_index()
    vars_df.index = range(1, len(prot_cols) + 1)
    vars_df.columns = ["col", "variants"]
    merged = pd.merge(variants_aln_info, vars_df, on = "col", how = "left")
    merged.index = range(1, len(vars_df) + 1)
    merged["shenkin"] = shenkin_aln["shenkin"]
    merged["rel_norm_shenkin"] = shenkin_aln["rel_norm_shenkin"] 
    merged["abs_norm_shenkin"] = shenkin_aln["abs_norm_shenkin"]
    if get_or == True:
        merged_or = get_OR(merged)
        return merged_or
    else:
        return merged

def generate_subset_aln(aln_in, aln_fmt, df, aln_out = None):
    """
    Creates a subset MSA containing only human sequences that present
    missense variants and returns the path of such MSA.
    """
    seqs_ids = df.source_id.unique().tolist()
    aln = Bio.SeqIO.parse(aln_in, aln_fmt)
    variant_seqs = [rec for rec in aln if rec.id in seqs_ids]
    n_variant_seqs = len(variant_seqs)
    if n_variant_seqs == 0:
        return ""
    else:
        log.info("There are {} sequences with variants for {}".format(str(n_variant_seqs), aln_in))
    if aln_out == None:
        aln_root, aln_ext = os.path.splitext(aln_in)
        aln_out =  "{}_variant_seqs{}".format(aln_root, aln_ext)
    Bio.SeqIO.write(variant_seqs, aln_out, aln_fmt)
    return aln_out

def get_OR(df, variant_col = "variants"):
    """
    Calculates OR, and associated p-value and CI,
    given a missense dataframe with variants and occupancy.
    """
    tot_occ = sum(df.occ)
    tot_vars = sum(df[variant_col])
    idx = df.index.tolist()
    for i in idx:
        i_occ = df.loc[i,"occ"]
        i_vars = df.loc[i,variant_col]
        rest_occ = tot_occ - i_occ
        rest_vars = tot_vars - i_vars
        if i_occ == 0:
            oddsr = np.nan 
            pval = np.nan
            se_or = np.nan
            log.debug("0 occupancy. Returning np.nan")
        else:
            if i_vars == 0:
                i_occ += 0.5
                i_vars += 0.5
                rest_occ += 0.5
                rest_vars += 0.5
                log.debug("0 variants. Adding 0.5 to each cell")
            oddsr, pval = stats.fisher_exact([[i_vars, rest_vars], [i_occ, rest_occ]])
            vals = [i_vars, rest_vars, i_occ, rest_occ]
            se_or = 1.96*(math.sqrt(sum(list(map((lambda x: 1/x), vals)))))
        df.loc[i, "oddsratio"] = round(oddsr, 2)
        df.loc[i, "pvalue"] = round(pval, 2)
        df.loc[i, "se_OR"] = round(se_or, 2)
    return df

def add_miss_class(df, miss_df_out = None, cons_col = "shenkin", MES_t = 1.0, cons_ts = [25, 75], colours = consvar_class_colours):
    """
    Adds two columns to missense dataframe. These columns will put columns
    into classes according to their divergence and missense enrichment score.
    It also adds a column that will help colour MSA columns according to their
    classifications.
    """
    for i in df.index:
        if df.loc[i, cons_col] <= cons_ts[0] and df.loc[i, "oddsratio"] < MES_t:
            df.loc[i, "miss_class"] = "CMD"
        elif df.loc[i, cons_col] <= cons_ts[0] and df.loc[i, "oddsratio"] > MES_t:
            df.loc[i, "miss_class"] = "CME"
        elif df.loc[i, cons_col] >= cons_ts[1] and df.loc[i, "oddsratio"] < MES_t:
            df.loc[i, "miss_class"] = "UMD"
        elif df.loc[i, cons_col] >= cons_ts[1] and df.loc[i, "oddsratio"] > MES_t:
            df.loc[i, "miss_class"] = "UME"
        else:
            df.loc[i, "miss_class"] = "None"
    coloring = {
        "CMD": colours[0],
        "CME": colours[1],
        "UMD": colours[3],
        "UME": colours[4],
        "None": colours[2]
    }
    df["miss_color"] =  df.miss_class.map(coloring)
    
    if miss_df_out != None:
        df.to_pickle(miss_df_out)
    return df

def merge_shenkin_df_and_mapping(shenkin_df, mapping_df, aln_ids):
    """
    merges conservation, and variation table with MSA-UniProt
    mapping table, so conservation and variation data
    are mapped to UniProt residues.
    """
    shenkin_df = shenkin_df.rename(index = str, columns = {"col": "alignment_column"}) # renaming columns to be consistent with other StruVarPi dataframes
    prot_mapping = mapping_df.copy(deep = True).loc[aln_ids]
    prot_mapping.columns = prot_mapping.columns.droplevel(1)
    prot_mapping.reset_index(inplace = True)
    prot_mapping = prot_mapping.rename_axis(None, axis = "columns")
    prot_mapping.rename(index = None, columns = {"Alignment": "MSA_column", "Protein_position": "UniProt_ResNum"}, inplace = True)
    mapped_data = pd.merge(
        prot_mapping[["MSA_column", "UniProt_ResNum"]], shenkin_df,
        left_on = "MSA_column", right_on = "alignment_column"
    ).drop("MSA_column", axis = 1)
    return mapped_data

def get_bss_table(results_df):
    all_bs_ress = results_df.query('binding_sites == binding_sites').reset_index(drop=True)
    all_bs_ress = all_bs_ress.explode("binding_sites")
    all_bs_ress["bs_id"] = all_bs_ress.job_id + "." + all_bs_ress.binding_sites.astype(str)
    all_bs_ress.UniProt_ResNum = all_bs_ress.UniProt_ResNum.astype(int)
    all_bs_ress["RSA"].values[all_bs_ress["RSA"].values > 100] = 100

    
    site_ids, site_rsas, site_shenks, site_mess, site_sizes = [[], [], [], [], []]
    for bs_id, bs_rows in all_bs_ress.groupby("bs_id"):            
        no_rsa = len(bs_rows.query('RSA != RSA'))
        no_shenk = len(bs_rows.query('abs_norm_shenkin != abs_norm_shenkin'))
        no_mes = len(bs_rows.query('oddsratio != oddsratio'))

        bs_rows = bs_rows.drop_duplicates(["binding_sites", "UniProt_ResNum"]) # drop duplicate residues within the binding site
        site_rsa = round(bs_rows.query('RSA == RSA').RSA.mean(),1)
        site_shenk = round(bs_rows.query('abs_norm_shenkin == abs_norm_shenkin').abs_norm_shenkin.mean(),1)
        site_mes = round(bs_rows.query('oddsratio == oddsratio').oddsratio.mean(),2)
        site_size = len(bs_rows)

        site_ids.append(bs_id.split(".")[-1])
        site_rsas.append(site_rsa)
        site_shenks.append(site_shenk)
        site_mess.append(site_mes)
        site_sizes.append(site_size)

    bss_data = pd.DataFrame(
        list(zip(site_ids, site_rsas, site_shenks, site_mess, site_sizes)),
        columns = ["lab", "RSA", "an_shenk", "MES", "n_ress"]
    )

    bss_data["Cluster"] = -1 # placeholder for cluster label (need to replace)
    bss_data["FS"] = -1 # placeholder for functional score (need to replace)

    return all_bs_ress, bss_data

### SETTING UP LOG

logging.basicConfig(filename = "LIGYSIS.log", format = '%(asctime)s %(name)s [%(levelname)-8s] - %(message)s', level = logging.INFO)

log = logging.getLogger("LIGYSIS")

### MAIN FUNCTION

def main(args):
    """
    Main function of the script. Calls all other functions.
    """

    ### PARSING ARGUMENTS

    log.info("Logging initiated")

    for arg, value in sorted(vars(args).items()):
        log.info("Argument %s: %r", arg, value)

    input_dir = args.input_dir
    uniprot_id = args.uniprot_id
    struc_fmt = args.struc_fmt
    OVERRIDE = args.override
    OVERRIDE_variants = args.override_variants
    run_variants = args.variants
    lig_clust_method = args.clust_method
    lig_clust_dist = args.clust_dist
    JACKHMMER_n_it = args.hmm_iters
    MES_t = args.mes_thresh
    cons_t_low =  args.cons_thresh_low
    cons_t_high = args.cons_thresh_high

    cons_ts = [cons_t_low, cons_t_high]

    ### SETTING UP DIRECTORIES

    input_id = os.path.normpath(input_dir).split(os.sep)[-1]
    main_output_dir = os.path.join(wd, "OUT")
    output_dir = os.path.join(main_output_dir, input_id)
    results_dir = os.path.join(output_dir, "results")
    raw_pdbs_dir = os.path.join(output_dir, "raw_pdbs")
    raw_cifs_dir = os.path.join(output_dir, "raw_cifs")
    stamp_out_dir = os.path.join(output_dir, "stamp_out")
    supp_pdbs_dir = os.path.join(output_dir, "supp_pdbs")
    supp_cifs_dir = os.path.join(output_dir, "supp_cifs")
    simple_cifs_dir = os.path.join(output_dir, "simple_cifs")
    clean_pdbs_dir = os.path.join(output_dir, "clean_pdbs")
    pdb_clean_dir = os.path.join(output_dir, "pdb_clean")
    mappings_dir = os.path.join(output_dir, "mappings")
    dssp_dir = os.path.join(output_dir, "dssp")
    arpeggio_dir = os.path.join(output_dir, "arpeggio")
    varalign_dir = os.path.join(output_dir, "varalign")
    
    dirs = [
        main_output_dir, output_dir, results_dir,
        raw_pdbs_dir, raw_cifs_dir,
        clean_pdbs_dir, stamp_out_dir,
        supp_pdbs_dir, supp_cifs_dir,
        simple_cifs_dir,
        pdb_clean_dir, mappings_dir, dssp_dir,
        arpeggio_dir, varalign_dir,
    ]
    
    setup_dirs(dirs) # creates directory /output/input_dir_name and then all results subdirectories: results, stamp_out, clean_pdbs, etc...

    log.info("Directories created")

    ### CHECKING IF FINAL RESULTS TABLE ALREADY EXISTS

    final_table_out = os.path.join(results_dir, "{}_results_table.pkl".format(input_id))
    if os.path.isfile(final_table_out) and not OVERRIDE:
        log.info("Final results table already exists")
        log.info("Skipping to the end")
        sys.exit(0)

    ### EXTRACTING UNIPROT PROTEIN ACCESSION, ENTRY AND NAME

    uniprot_info_out = os.path.join(results_dir, "{}_uniprot_info.pkl".format(input_id))
    if OVERRIDE or not os.path.isfile(uniprot_info_out):
        if uniprot_id != "unknown": # PLACEHOLDER FOR WHEN THERE IS NO UNIPROT ID PROVIDED BY THE USER
            prot_info = get_uniprot_info(uniprot_id)
            dump_pickle(prot_info, uniprot_info_out)
            log.info("Obtained UniProt protein names")
        else:
            prot_info = {"up_id": uniprot_id, "up_entry": "", "prot_name": ""}
            dump_pickle(prot_info, uniprot_info_out)
            log.info("UniProt ID was not provided, so there are no protein names...")
    else:
        pass

    ### GETTING STRUCTURES

    if struc_fmt == "mmcif":
        strucs = [f for f in os.listdir(input_dir) if f.endswith(".cif")] # different extension depending on structure format
    elif struc_fmt == "pdb":
        strucs = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]
        if len(strucs) == 0:
            strucs = [f for f in os.listdir(input_dir) if f.endswith(".ent")] # try .ent (this is default extension of PDB files downloaded from PDBe)
    n_strucs = len(strucs)
    log.info("Number of structures: {}".format(n_strucs))

    ### If input format is mmcif, we need to convert to pdb since clean_pdb.py only works with pdb files

    if struc_fmt == "mmcif":
        for struc in strucs:
            struc_root, struc_ext = os.path.splitext(struc)
            log.info("Converting {} to pdb".format(struc_root))
            input_cif_path = os.path.join(input_dir, struc)
            shutil.copy(input_cif_path, raw_cifs_dir) # copy all structures to raw_cifs_dir and then transform format to .pdb and save to raw_pdbs_dir

            cif_df = PDBXreader(input_cif_path).atoms(format_type = struc_fmt, excluded=())
            pdb_out = os.path.join(raw_pdbs_dir, "{}.pdb".format(struc_root))

            ### replacing label_seq_id by auth_seq_id to cif_df, needed so HETATM have resnums
            cif_df["label_seq_id"] = cif_df["auth_seq_id"]

            w = PDBXwriter(outputfile = pdb_out)
            w.run(cif_df, format_type = "pdb")
            log.info("Converted {} to pdb".format(struc))
    else:
        for struc in strucs:
            struc_root, struc_ext = os.path.splitext(struc)
            input_pdb_path = os.path.join(input_dir, struc)
            output_pdb_path = os.path.join(raw_pdbs_dir, struc)
            if OVERRIDE or not os.path.isfile(output_pdb_path):
                shutil.copy(input_pdb_path, raw_pdbs_dir)
                log.info("Copied {} to raw_pdbs_dir".format(struc))
            else:
                log.debug("File {} already exists".format(output_pdb_path))


    ### CLEANING FILES

    pdb_strucs = [f for f in os.listdir(raw_pdbs_dir) if f.endswith(".pdb") and "clean" not in f]
    if pdb_strucs == []:
        pdb_strucs = [f for f in os.listdir(raw_pdbs_dir) if f.endswith(".ent") and "clean" not in f] # if no .pdb files, try .ent files

    for struc in pdb_strucs:
        struc_root, struc_ext = os.path.splitext(struc)  # "structure_name", ".pdb"
        pdb_path = os.path.join(raw_pdbs_dir, struc)        # /output_dir/raw_pdbs/structure_name.pdb
        pdb_path_root, _ =  os.path.splitext(pdb_path)   # /output_dir/raw_pdbs/structure_name
        clean_struc_path_from = f'{pdb_path_root}.clean{struc_ext}' # /input_dir/structure_name.clean.pdb
        clean_struc_path_to = os.path.join(clean_pdbs_dir, f'{struc_root}.clean{struc_ext}') # /output_dir/clean_pdbs/structure_name.clean.pdb
        if os.path.isfile(clean_struc_path_to): # if clean file already exists in clean files dir
            log.debug("PDB {} already cleaned".format(struc))
            pass
        else: # if clean file does not exist in clean files dir
            cmd, ec = run_clean_pdb(pdb_path) # run pdb_clean.py on pdb file
            if ec == 0:
                log.debug("{} cleaned".format(struc))
                if os.path.isfile(clean_struc_path_from): 
                    shutil.move(clean_struc_path_from, clean_struc_path_to) # move clean file to clean files dir
                for pdb_clean_suff in pdb_clean_suffixes: # for each suffix in pdb_clean_suffixes
                    pdb_clean_file_from = os.path.join(raw_pdbs_dir, "{}.{}".format(struc, pdb_clean_suff))
                    pdb_clean_file_to = os.path.join(pdb_clean_dir, "{}.{}".format(struc, pdb_clean_suff))
                    if os.path.isfile(pdb_clean_file_from):
                        shutil.move(pdb_clean_file_from, pdb_clean_file_to)
            else:
                log.critical("pdb_clean.py failed with {}".format(cmd))

    ### STAMP SECTION

    domains_out = os.path.join(results_dir, "{}_stamp.domains".format(input_id))
    if os.path.isfile(domains_out):
        log.debug("STAMP domains file already exists")
        pass
    else:
        generate_STAMP_domains(wd, clean_pdbs_dir, domains_out)
        log.info("STAMP domains file generated")

    prefix = "{}_stamp".format(input_id)
    n_domains = len(open(domains_out).readlines())
    log.info("Number of domains: {}".format(str(n_domains)))
    matrix_file = "{}.{}".format(prefix, str(n_domains-1))
    last_matrix_path = os.path.join(stamp_out_dir, matrix_file)
    fnames = fnames_from_domains(domains_out) # these thould be stamped files, so with .supp

    ### DO NOT RUN STAMP IF INPUT IS A SINGLE CHAIN ###
    #print(os.listdir(clean_pdbs_dir))
    if n_domains == 1: # single structure
        #pass
        clean_file = os.listdir(clean_pdbs_dir)[0]
        shutil.move(os.path.join(clean_pdbs_dir, clean_file), os.path.join(wd, fnames[0])) # just copy clean pdb and change name
    else:
        if os.path.isfile(last_matrix_path):
            log.debug("STAMP matrix files already exist")
            pass
        else:
            cmd, ec = stamp(
                domains_out,
                prefix, os.path.join(results_dir, "{}.out".format(prefix))
            )
            if ec == 0:
                log.info("STAMP matrix files generated")
            else:
                log.critical("STAMP failed with {}".format(cmd))
            
            c = 0 # counting the number of superposed pdbs
            for file in fnames:
                if os.path.isfile(os.path.join(supp_pdbs_dir, file)): # only when they alaready have been transformed
                    c += 1
            if c == n_domains:
                log.debug("All structure domains already are superposed")
                pass
            else:
                if not os.path.isfile(matrix_file): # RUNNING TRANSFORM ONCE STAMP OUTPUT HAS BEEN MOVED TO STAMP_OUT_DIR
                    matrix_file = os.path.join(stamp_out_dir, matrix_file) # needs to change to accommodate for how many domains in .domains file 
                cmd, ec = transform(matrix_file) #running transform with matrix on cwd
                if ec == 0:
                    log.info("Structures transformed")
                else:
                    log.critical("TRANSFORM failed with {}".format(cmd))
            
            log.info("STAMP and TRANSFORM completed")
    
    move_supp_files(fnames, supp_pdbs_dir, wd)

    move_stamp_output(prefix, stamp_out_dir, wd)

    ### CONVERT SUPERPOSED FILES TO MMCIF FORMAT

    supp_pdbs = [f for f in os.listdir(supp_pdbs_dir) if f.endswith(".pdb")]
    for file in supp_pdbs:
        file_root, _ = os.path.splitext(file)
        supp_file = os.path.join(supp_pdbs_dir, file)
        cif_file = os.path.join(supp_cifs_dir, "{}.cif".format(file_root))
        if os.path.isfile(cif_file):
            log.debug(f"{cif_file} file already exists")
            pass
        else:
            pdb_df = PDBXreader(supp_file).atoms(format_type = "pdb", excluded=())
            pdb_df = add_double_quotes_for_single_quote(pdb_df)

            pdb_df["label_alt_id"] = "."
            pdb_df["pdbx_formal_charge"] = "?"

            w = PDBXwriter(outputfile = cif_file)
            w.run(pdb_df[cif_cols_order], format_type = "mmcif")
            log.info("Converted {} to mmcif".format(file_root))
    log.info("Superposed files have been converted to mmcif format")

    ### CIF SIMPLIFICATION SECTION

    simple_cifs = [f for f in os.listdir(simple_cifs_dir) if f.endswith(".cif")]
    n_simple_cifs = len(simple_cifs) # number of simplified pdbs, will be 0 the first time thiis command is executed

    if n_simple_cifs == n_domains:
        log.debug("Structure domains already simplified")
        pass
    else:
        supp_files = sorted([f for f in os.listdir(supp_cifs_dir) if f.endswith(".cif")])
        cif_root, cif_ext = os.path.splitext(supp_files[0])
        cif_name = cif_root.replace(".supp", "") # can't rely on file names not having "." in them
        simple_file_name = f'{cif_name}.simp{cif_ext}'
        shutil.copy(os.path.join(supp_cifs_dir, supp_files[0]), os.path.join(simple_cifs_dir, simple_file_name)) # copy first chain as is
        log.info(f'Keeping protein atoms for {supp_files[0]}')
        for file in supp_files[1:]: # we keep protei atoms for first chain
            if file.endswith(".cif"):
                supp_file = os.path.join(supp_cifs_dir, file)
                file_root, file_ext = os.path.splitext(file)
                file_name = file_root.replace(".supp", "") # can't rely on file names not having "." in them
                simple_file = os.path.join(simple_cifs_dir, f'{file_name}.simp{file_ext}')
                if os.path.isfile(simple_file):
                    log.debug("Simple pdb file already exists")
                    pass
                else:
                    simplify_pdb(supp_file, simple_file, "mmcif")
        log.info("All structure domains have been simplified") # what we want to do here is seimply keep protein atoms for first chain, this is to make visualisation quicker and simpler
    
    log.info("CIF simplification completed")

    ### UNIPROT MAPPING SECTION

    supp_strucs = [f for f in os.listdir(supp_cifs_dir) if f.endswith(".cif")]

    for struc in supp_strucs: #fnames are now the files of the STAMPED PDB files, not the original ones
        struc_root, _ =  os.path.splitext(struc)
        struc_name = struc_root.replace(".supp", "") # can't rely on file names not having "." in them
        struc_mapping_path = os.path.join(mappings_dir, "{}_mapping.csv".format(struc_name))
        pdb2up_mapping_dict_path = os.path.join(mappings_dir, "{}_pdb2up.pkl".format(struc_name))
        up2pdb_mapping_dict_path = os.path.join(mappings_dir, "{}_up2pdb.pkl".format(struc_name))
        if OVERRIDE or not os.path.isfile(struc_mapping_path):
            if uniprot_id != "unknown" and prot_info["up_id"] != "":
                mapping = retrieve_mapping_from_struc(struc, uniprot_id, supp_cifs_dir, mappings_dir, struc_fmt = "mmcif") # giving supp, here, instead of simple because we want them all
                mapping_dict, up2pdb = create_resnum_mapping_dicts(mapping)
                dump_pickle(mapping_dict, pdb2up_mapping_dict_path)
                dump_pickle(up2pdb, up2pdb_mapping_dict_path)
                log.info("Mapping files for {} generated".format(struc_name))
            else: # if no UniPrt ID is provided. Generate psuedo-mapping. Will be problematic if all structures don't share the same numbering
                mapping = get_pseudo_mapping_from_struc(struc, supp_cifs_dir, mappings_dir, struc_fmt = "mmcif")
                mapping_dict, up2pdb = create_resnum_mapping_dicts(mapping)
                dump_pickle(mapping_dict, pdb2up_mapping_dict_path)
                dump_pickle(up2pdb, up2pdb_mapping_dict_path)
                log.info("Pseudo-mapping files for {} generated".format(struc_name))
        else:
            log.debug("Mapping files for {} already exists".format(struc_root))

    log.info("UniProt mapping section completed")
            
    ### DSSP SECTION   

    dssp_mapped_out = os.path.join(results_dir, "{}_dssp_mapped.pkl".format(input_id))

    if OVERRIDE or not os.path.isfile(dssp_mapped_out):

        mapped_dssps = []
        for struc in fnames: #fnames are now the files of the STAMPED PDB files, not the original ones
            struc_root, _ =  os.path.splitext(struc)
            struc_name = struc_root.replace(".supp", "") # can't rely on file names not having "." in them
            dssp_csv = os.path.join(dssp_dir, "{}.csv".format(struc_name))

            if OVERRIDE or not os.path.isfile(dssp_csv):
                dssp_data = run_dssp(struc, supp_pdbs_dir, dssp_dir)
                log.info("DSSP run successfully on {}".format(struc_name))
            else:
                dssp_data = pd.read_csv(dssp_csv)
                log.debug("DSSP data already exists")
            
            ## UNIPROT MAPPING
            struc_mapping_path = os.path.join(mappings_dir, "{}_mapping.csv".format(struc_name))
            mapping = pd.read_csv(struc_mapping_path)

            dssp_data.PDB_ResNum = dssp_data.PDB_ResNum.astype(str)
            mapping.PDB_ResNum = mapping.PDB_ResNum.astype(str)
            mapping = pd.merge(mapping, dssp_data, left_on = "PDB_ResNum", right_on = "PDB_ResNum") # don't think this merging worked well

            mapped_dssps.append(mapping)

        mapped_dssp_df = pd.concat(mapped_dssps)

        mapped_dssp_df.to_pickle(os.path.join(results_dir, "{}_dssp_mapped.pkl".format(input_id)))
    else:
        mapped_dssp_df = pd.read_pickle(dssp_mapped_out)
        log.debug("Loaded mapped DSSP data")

    log.info("DSSP section completed")

    ### GET LIGAND DATA

    lig_data_path = os.path.join(results_dir, "{}_lig_data.pkl".format(input_id))
    if OVERRIDE or not os.path.isfile(lig_data_path):
        ligs_df = get_lig_data(supp_cifs_dir, lig_data_path, "mmcif") # this now uses auth fields as it seems that is what pdbe-arpeggio uses.
        log.info("Saved ligand data")
    else:
        ligs_df = pd.read_pickle(lig_data_path)
        log.debug("Loaded ligand data")

    ### ARPEGGIO PART ###

    fps_out = os.path.join(results_dir, f'{input_id}_ligs_fingerprints.pkl') 
    fps_status_out = os.path.join(results_dir, f'{input_id}_fps_status.pkl') #fps: will stand for fingerprints

    if OVERRIDE or not os.path.isfile(fps_out) or not os.path.isfile(fps_status_out):
            
        struc2ligs = {}
        lig_fps = {}
        fp_status = {}

        for struc in supp_strucs:
            struc_root, _ =  os.path.splitext(struc)
            struc_name = struc_root.replace(".supp", "") # can't rely on file names not having "." in them

            struc2ligs[struc] = []

            struc_df = ligs_df.query('struc_name == @struc')

            struc_path = os.path.join(supp_cifs_dir, struc)
            
            if struc_df.empty:
                fp_status[struc_root] = "No-Ligs"
                log.warning("No ligands in {}".format(struc))
                continue

            lig_sel = " ".join(["/{}/{}/".format(row.auth_asym_id, row.auth_seq_id) for _, row in  struc_df.iterrows()])

            ligand_names = list(set([row.label_comp_id for _, row in struc_df.iterrows()]))

            arpeggio_default_json_name = os.path.basename(struc_name)
            arpeggio_default_out = os.path.join(arpeggio_dir, f"{arpeggio_default_json_name}.json")  # this is how arpeggio names the file (splits by "." and takes the first part)

            arpeggio_out = os.path.join(arpeggio_dir, struc_name + ".json")
            arpeggio_proc_df_out = os.path.join(arpeggio_dir, struc_name + "_proc.pkl")

            if OVERRIDE or not os.path.isfile(arpeggio_proc_df_out):

                if OVERRIDE or not os.path.isfile(arpeggio_out):

                    ec, cmd = run_arpeggio(struc_path, lig_sel, arpeggio_dir)
                    if ec != 0:
                        fp_status[struc_name] = "Arpeggio-Fail"
                        log.error("Arpeggio failed for {} with {}".format(struc, cmd))
                        continue

                    shutil.move(arpeggio_default_out, arpeggio_out)

                arp_df = pd.read_json(arpeggio_out)

                pdb2up = load_pickle(os.path.join(mappings_dir, "{}_pdb2up.pkl".format(struc_name)))

                proc_inters, fp_stat = process_arpeggio_df(arp_df, struc_name, ligand_names, pdb2up)

                coords_dict = generate_dictionary(struc_path)

                proc_inters["coords_end"] = proc_inters.set_index(["auth_asym_id_end", "label_comp_id_end", "auth_seq_id_end", "auth_atom_id_end"]).index.map(coords_dict.get)
                proc_inters["coords_bgn"] = proc_inters.set_index(["auth_asym_id_bgn", "label_comp_id_bgn", "auth_seq_id_bgn", "auth_atom_id_bgn"]).index.map(coords_dict.get)

                proc_inters["width"] = proc_inters["contact"].apply(determine_width)
                proc_inters["color"] = proc_inters["contact"].apply(determine_color)

                proc_inters["width"] = proc_inters["width"]
                proc_inters.to_pickle(arpeggio_proc_df_out)

                fp_status[struc_root] = fp_stat
                log.debug("Arpeggio processed data saved")
            else:
                proc_inters = pd.read_pickle(arpeggio_proc_df_out)
                log.debug("Loaded arpeggio processed data")

            proc_inters_indexed = proc_inters.set_index(["label_comp_id_bgn", "auth_asym_id_bgn", "auth_seq_id_bgn"])

            lig_fps_status = {}
            for _, row in  struc_df.iterrows():
                lig = (row.label_comp_id, row.auth_asym_id, row.auth_seq_id)
                try:
                    lig_rows = proc_inters_indexed.loc[[lig], :].copy()  # Happens for 7bf3 (all aa binding MG are artificial N-term), also for low-occuoancy ligands? e.g., 5srs, 5sq5

                    if lig_rows.isnull().values.all(): # need all so works only when WHOLE row is Nan
                        log.warning("No interactions for ligand {} in {}".format(lig, struc_root))
                        lig_fps_status[lig] = "No-PLIs"
                        continue

                    ###### CHECK IF LIGAND FINGERPRINT IS EMPTY ######
                    lig_rows.UniProt_ResNum_end = lig_rows.UniProt_ResNum_end.astype(int)
                    lig_fp = lig_rows.UniProt_ResNum_end.unique().tolist()
                    lig_key = "{}_".format(struc_name) + "_".join([str(l) for l in lig]) # can't rely on file names not having "." in them
                    
                    lig_fps[lig_key] = lig_fp
                except:
                    log.warning("Empty fingerprint for ligand {} in {}".format(lig, struc_root))
                    continue
        dump_pickle(lig_fps, fps_out)
        dump_pickle(fp_status, fps_status_out)
    else:
        lig_fps = pd.read_pickle(fps_out)
        fp_status = pd.read_pickle(fps_status_out)
        log.debug("Loaded fingerprints")
    
    log.info("ARPEGGIO section completed")

    ### BINDING SITE DEFINITION SECTION NEW

    lig_inters = get_inters(lig_fps)
    lig_inters = [sorted(list(set(i))) for i in lig_inters] # making sure each residue is present only once (O00214 problematic with saccharides)
    lig_labs = get_labs(lig_fps)
    n_ligs = len(lig_labs)
    log.info("There are {} relevant ligands for {}".format(str(n_ligs), input_id))
    irel_mat_out = os.path.join(results_dir, f'{input_id}_irel_matrix.pkl')

    #### GENERATING IREL MATRIX

    if OVERRIDE or not os.path.isfile(irel_mat_out):
        irel_matrix = get_intersect_rel_matrix(lig_inters) # this is a measure of similarity, probs want to save this
        dump_pickle(irel_matrix, irel_mat_out)
        log.info("Calcualted intersection matrix")
    else:
        irel_matrix = load_pickle(irel_mat_out)
        log.debug("Loaded intersection matrix")

    #### CLUSTERING LIGANDS

    if n_ligs == 1:
        cluster_ids = [0]
    else:
        irel_df = pd.DataFrame(irel_matrix)
        dist_df = 1 - irel_df # distance matrix in pd.Dataframe() format
        condensed_dist_mat = scipy.spatial.distance.squareform(dist_df) # condensed distance matrix to be used for clustering
        linkage = scipy.cluster.hierarchy.linkage(condensed_dist_mat, method = lig_clust_method, optimal_ordering = True)
        cut_tree = scipy.cluster.hierarchy.cut_tree(linkage, height = lig_clust_dist)
        cluster_ids = [int(cut) for cut in cut_tree]
    
    cluster_id_dict = {lig_labs[i]: cluster_ids[i] for i in range(len(lig_labs))} #dictionary indicating membership for each lig
            
    log.info(f'Ligand clustering realised for {input_id}')

    #### GENERATING CHIMERAX FILES

    attr_out = os.path.join(simple_cifs_dir,f'{input_id}_{lig_clust_method}_{lig_clust_dist}.defattr')
    chimera_script_out = os.path.join(simple_cifs_dir,f'{input_id}_{lig_clust_method}_{lig_clust_dist}.cxc')
    chX_session_out = os.path.join(simple_cifs_dir,f'{input_id}_{lig_clust_method}_{lig_clust_dist}.cxs')

    if OVERRIDE or not os.path.isfile(attr_out):
                
        order_dict = write_chimeraX_attr(cluster_id_dict, simple_cifs_dir, attr_out) # this actually needs to be simplified PDBs, not transformed ones ???

    if OVERRIDE or not os.path.isfile(chimera_script_out) or not os.path.isfile(chX_session_out):

        write_chimeraX_script(chimera_script_out, simple_cifs_dir, os.path.basename(attr_out), os.path.basename(chX_session_out), chimeraX_commands, cluster_ids) # this actually needs to be simplified PDBs, not transformed ones ???

    log.info(f'Chimera attributes and script generated for {input_id}')

    log.info("Binding site definition completed")

    ### BINDING SITE MEMBERSHIP PROCESSING

    membership_out = os.path.join(results_dir, f"{input_id}_bss_membership.pkl")
    cluster_ress_out = os.path.join(results_dir, f"{input_id}_bss_ress.pkl")
    bs_mm_dict_out = os.path.join(results_dir, f"{input_id}_ress_bs_membership.pkl")
        
    if OVERRIDE or not os.path.isfile(membership_out):
        membership = get_cluster_membership(cluster_id_dict) # which LBS ligands belong to
        dump_pickle(membership, membership_out)
        log.info("Calculated binding site membership")
    else:
        membership = load_pickle(membership_out)
        log.debug("Loaded binding site membership")

    if OVERRIDE or not os.path.isfile(cluster_ress_out):
        cluster_ress = get_all_cluster_ress(membership, lig_fps) # residues that form each LBS 
        dump_pickle(cluster_ress, cluster_ress_out) 
        log.info("Calculated binding site composition") 
    else:
        cluster_ress = load_pickle(cluster_ress_out)
        log.debug("Loaded binding site composition")

    if OVERRIDE or not os.path.isfile(bs_mm_dict_out):
        bs_ress_membership_dict = get_residue_bs_membership(cluster_ress)
        log.info("Calcualted residue membership")
        dump_pickle(bs_ress_membership_dict, bs_mm_dict_out)  
    else:
        bs_ress_membership_dict = load_pickle(bs_mm_dict_out)
        log.debug("Loaded residue membership")

    ### DSSP DATA ANALYSIS
    mapped_dssp_df['RSA'].replace("", np.nan, inplace=True)
    dsspd_filt = mapped_dssp_df.query('UniProt_ResNum == UniProt_ResNum and PDB_ResName != "X" and RSA == RSA').copy()
    dsspd_filt.SS = dsspd_filt.SS.fillna("C")
    dsspd_filt.SS = dsspd_filt.SS.replace("", "C")
    dsspd_filt.UniProt_ResNum = dsspd_filt.UniProt_ResNum.astype(int)

    AA_dict_out = os.path.join(results_dir, "{}_ress_AA.pkl".format(input_id))
    RSA_dict_out = os.path.join(results_dir, "{}_ress_RSA.pkl".format(input_id))
    SS_dict_out = os.path.join(results_dir, "{}_ress_SS.pkl".format(input_id))
    rsa_profs_out = os.path.join(results_dir, "{}_bss_RSA_profiles.pkl".format(input_id))
    ss_profs_out = os.path.join(results_dir, "{}_bss_SS_profiles.pkl".format(input_id))
    aa_profs_out = os.path.join(results_dir, "{}_bss_AA_profiles.pkl".format(input_id))

    if OVERRIDE or not os.path.isfile(AA_dict_out):
        ress_AA_dict = {
            up_resnum: dsspd_filt.query('UniProt_ResNum == @up_resnum').PDB_ResName.mode()[0] # gets dict per UP residue and more frequent AA.
            for up_resnum in dsspd_filt.UniProt_ResNum.unique().tolist()
        }
        dump_pickle(ress_AA_dict, AA_dict_out)
        log.info("Saved AA dict")
    else:
        ress_AA_dict = load_pickle(AA_dict_out)
        log.debug("Loaded AA dict")
    
    if OVERRIDE or not os.path.isfile(RSA_dict_out):
        ress_RSA_dict = {
            up_resnum: round(dsspd_filt.query('UniProt_ResNum == @up_resnum').RSA.mean(), 2) # gets dict per UP residue and average RSA.
            for up_resnum in dsspd_filt.UniProt_ResNum.unique().tolist()
        }
        dump_pickle(ress_RSA_dict, RSA_dict_out)
        log.info("Saved RSA dict")
    else:
        ress_RSA_dict = load_pickle(RSA_dict_out)
        log.debug("Loaded RSA dict")

    if OVERRIDE or not os.path.isfile(SS_dict_out):
        ress_SS_dict = {
            up_resnum: dsspd_filt.query('UniProt_ResNum == @up_resnum').SS.mode()[0] # gets dict per UP residue and more frequent SS.
            for up_resnum in dsspd_filt.UniProt_ResNum.unique().tolist()
        }
        dump_pickle(ress_SS_dict, SS_dict_out)
        log.info("Saved SS dict")
    else:
        ress_SS_dict = load_pickle(SS_dict_out)
        log.debug("Loaded SS dict")

    if OVERRIDE or not os.path.isfile(rsa_profs_out):
        rsa_profiles = {}
        for k, v in cluster_ress.items():
            rsa_profiles[k] = []
            for v2 in v:
                if v2 in ress_RSA_dict:
                    rsa_profiles[k].append(ress_RSA_dict[v2])
                else:
                    log.warning("Cannot find RSA data for UP residue {} in {}".format(str(v2), input_id))
        dump_pickle(rsa_profiles, rsa_profs_out)
        log.info("Saved RSA profiles")
    else:
        rsa_profiles = load_pickle(rsa_profs_out)
        log.debug("Loaded RSA profiles")
    
    if OVERRIDE or not os.path.isfile(ss_profs_out):
        ss_profiles = {}
        for k, v in cluster_ress.items():
            ss_profiles[k] = []
            for v2 in v:
                if v2 in ress_SS_dict:
                    ss_profiles[k].append(ress_SS_dict[v2])
                else:
                    log.warning("Cannot find SS data for UP residue {} in {}".format(str(v2), input_id))
        dump_pickle(ss_profiles, ss_profs_out)
        log.info("Saved SS profiles")
    else:
        ss_profiles = load_pickle(ss_profs_out)
        log.debug("Loaded SS profiles")

    if OVERRIDE or not os.path.isfile(aa_profs_out):
        aa_profiles = {}
        for k, v in cluster_ress.items():
            aa_profiles[k] = []
            for v2 in v:
                if v2 in ress_AA_dict:
                    aa_profiles[k].append(ress_AA_dict[v2])
                else:
                    log.warning("Cannot find AA data for UP residue {} in {}".format(str(v2), input_id))
        dump_pickle(aa_profiles, aa_profs_out)
        log.info("Saved AA profiles")
    else:
        aa_profiles = load_pickle(aa_profs_out)
        log.debug("Loaded AA profiles")

    log.info("DSSP analysis completed")

    ### VARIATION SECTION

    if run_variants: 

        example_struc = os.path.join(supp_cifs_dir, sorted([f for f in os.listdir(supp_cifs_dir) if f.endswith(".cif")])[0]) # first structure in the list, which is one with protein atoms on simple.
        struc_root, _ = os.path.splitext(os.path.basename(example_struc))
        struc_name = struc_root.replace(".supp", "") # can't rely on file names not having "." in them
        struc_mapping = load_pickle(os.path.join(mappings_dir, "{}_pdb2up.pkl".format(struc_name)))
        fasta_path = os.path.join(varalign_dir, "{}.fa".format(input_id))
        fasta_root, _ = os.path.splitext(fasta_path)    
        hits_aln = "{}.sto".format(fasta_root)  
        hits_aln_rf = "{}_rf.sto".format(fasta_root)
        shenkin_out = os.path.join(varalign_dir, "{}_shenkin.csv".format(input_id))
        shenkin_filt_out = os.path.join(varalign_dir, "{}_shenkin_filt.csv".format(input_id))

        if OVERRIDE_variants or not os.path.isfile(hits_aln_rf):
            create_alignment_from_struc(example_struc, fasta_path,  struc_fmt = "mmcif", n_it = JACKHMMER_n_it, seqdb = swissprot_path) # mmcif for supp cifs as need to be transofrmed for Arpeggio
            log.info("Saved MSA to file")
            pass
        else:
            log.debug("Loaded MSA from file")

        ### CONSERVATION ANALYSIS

        prot_cols = get_target_prot_cols(hits_aln)
        
        shenkin_out = os.path.join(varalign_dir, "{}_rf_shenkin.pkl".format(input_id))
        if OVERRIDE_variants or not os.path.isfile(shenkin_out):
            shenkin = calculate_shenkin(hits_aln_rf, "stockholm", shenkin_out)
            log.info("Saved conservation data table")
        else:
            shenkin = pd.read_pickle(shenkin_out)
            log.debug("Loaded conservation data table")
        
        shenkin_filt_out = os.path.join(varalign_dir, "{}_rf_shenkin_filt.pkl".format(input_id))
        if OVERRIDE_variants or not os.path.isfile(shenkin_filt_out):
            shenkin_filt = format_shenkin(shenkin, prot_cols, shenkin_filt_out)
            log.info("Saved filtered conservation data table")
        else:
            shenkin_filt = pd.read_pickle(shenkin_filt_out)
            log.debug("Loaded filtered conservation data table")

        aln_obj = Bio.AlignIO.read(hits_aln_rf, "stockholm") #crashes if target protein is not human!
        aln_info_path = os.path.join(varalign_dir, "{}_rf_info_table.p.gz".format(input_id))
        if OVERRIDE_variants or not os.path.isfile(aln_info_path):
            example_struc_df = PDBXreader(example_struc).atoms(format_type = "mmcif", excluded=())
            chains = sorted(list(example_struc_df.auth_asym_id.unique())) ### TODO: we generate sequence only for the first chain. this means this will only work for MONOMERIC PROTEINS
            main_chain = chains[0]
            pdb_resnums = list(struc_mapping[main_chain].keys())
            aln_info = varalign.alignments.alignment_info_table_FRAGSYS_CUSTOM(aln_obj, struc_mapping[main_chain]) ##### ADD PDB RESNUMS HERE OF STRUCTURE SEQUENCE
            aln_info.to_pickle(aln_info_path)
            log.info("Saved MSA info table")
        else:
            aln_info = pd.read_pickle(aln_info_path)
            log.debug("Loaded MSA info table")
        
        log.info("There are {} sequences in MSA".format(len(aln_info)))
        
        indexed_mapping_path = os.path.join(varalign_dir, "{}_rf_mappings.p.gz".format(input_id))
        if OVERRIDE_variants or not os.path.isfile(indexed_mapping_path):
            indexed_mapping_table = varalign.align_variants._mapping_table(aln_info) # now contains all species
            indexed_mapping_table.to_pickle(indexed_mapping_path) # important for merging later on
            log.info("Saved MSA mapping table")
        else:
            indexed_mapping_table = pd.read_pickle(indexed_mapping_path)
            log.debug("Loaded MSA mapping table")    

        aln_info_human = aln_info.query('species == "HUMAN"')

        if len(aln_info_human) > 0:
            log.info("There are {} HUMAN sequences in the MSA".format(len(aln_info_human)))

            human_hits_msa = os.path.join(varalign_dir, "{}_rf_human.sto".format(input_id))

            if OVERRIDE_variants or not os.path.isfile(human_hits_msa):
                get_human_subset_msa(hits_aln_rf, human_hits_msa)
            else:
                pass
           
            cp_path = cp_sqlite(wd)  ### copy ensemble SQLite to directory where this is being executed
            log.debug("ENSEMBL_CACHE SQLite copied correctly")

            variant_table_path = os.path.join(varalign_dir, "{}_rf_human_variants.p.gz".format(input_id))
            if OVERRIDE_variants or not os.path.isfile(variant_table_path):
                try:
                    variants_table = varalign.align_variants.align_variants(aln_info_human, path_to_vcf = gnomad_vcf,  include_other_info = False, write_vcf_out = False)     
                except ValueError as e:
                    variants_table = pd.DataFrame()
                    log.warning("No variants were retrieved")

                variants_table.to_pickle(variant_table_path)

            else:
                variants_table = pd.read_pickle(variant_table_path)

            
            rm_sqlite(cp_path) ### remove ensembl SQLite from directory where this is being executed
            log.debug("ENSEMBL_CACHE SQLite removed correctly")

            if variants_table.empty: # variant table is empty. E.g., P03915. Only 3 human sequences. They are all mitochondrial (not in gnomAD)
                pass

            else:
                # in order to be able to read the vcf and parse the DB, the ensemble.cache.sqlite file must be in the ./.varalign directory

                human_miss_vars = format_variant_table(variants_table, prot_cols) # GET ONLY MISSENSE VARIANTS ROWS
                human_miss_vars_msa_out = os.path.join(varalign_dir, "{}_rf_human_missense_variants_seqs.sto".format(input_id))

                miss_df_out = os.path.join(results_dir, "{}_missense_df.pkl".format(input_id))
                
                if OVERRIDE or not os.path.isfile(miss_df_out): # we leave it as OVERRIDE and not OVERRIDE_variants to fix the wrong pseudocounts
                    missense_variants_df = get_missense_df(
                        hits_aln_rf, human_miss_vars,
                        shenkin_filt, prot_cols, human_miss_vars_msa_out
                    )

                    if missense_variants_df.empty:
                        log.warning("No missense variants found for MSA")
                        pass

                    else:
                        missense_variants_df = add_miss_class(
                            missense_variants_df, miss_df_out,
                            cons_col = "abs_norm_shenkin", MES_t = MES_t, cons_ts = cons_ts,
                        )
                        log.info("Saved missense dataframe")
                else:
                    missense_variants_df = pd.read_pickle(miss_df_out)
                    log.debug("Loaded missense dataframe")

                if missense_variants_df.empty:
                    log.warning("No missense variants found for MSA of {}".format(input_id))
                    pass
                else:
                    # ADDS COLUMNS FROM MISSENSE DF TO SHENKIN FILT DF, CONSERVATION AND VARIATION DATA ABOUT HUMAN VARIANT SUB MSA
                    shenkin_filt.loc[:, "human_shenkin"] = missense_variants_df.shenkin
                    shenkin_filt.loc[:, "human_occ"] = missense_variants_df.occ
                    shenkin_filt.loc[:, "human_gaps"] = missense_variants_df.gaps
                    shenkin_filt.loc[:, "human_occ_pct"] = missense_variants_df.occ_pct
                    shenkin_filt.loc[:, "human_gaps_pct"] = missense_variants_df.gaps_pct
                    shenkin_filt.loc[:, "variants"] = missense_variants_df.variants
                    shenkin_filt.loc[:, "oddsratio"] = missense_variants_df.oddsratio
                    shenkin_filt.loc[:, "pvalue"] = missense_variants_df.pvalue
                    shenkin_filt.loc[:, "se_OR"] = missense_variants_df.se_OR

        else:
            log.warning("No human sequences in MSA")
            pass

        shenkin_mapped_out = os.path.join(results_dir, "{}_ress_consvar.pkl".format(input_id))
        if OVERRIDE or not os.path.isfile(shenkin_mapped_out): # we leave it as OVERRIDE and not OVERRIDE_variants to fix the wrong pseudocounts
            aln_ids = list(set([seqid[0] for seqid in indexed_mapping_table.index.tolist() if "query" in seqid[0]])) # THIS IS EMPTY IF QUERY SEQUENCE IS NOT FOUND
            n_aln_ids = len(aln_ids)
            if n_aln_ids != 1:
                log.warning("There are {} sequences matching input protein accession".format(str(n_aln_ids)))

            mapped_data = merge_shenkin_df_and_mapping(shenkin_filt, indexed_mapping_table, aln_ids)
            mapped_data.to_pickle(shenkin_mapped_out)
        else:
            mapped_data = pd.read_pickle(shenkin_mapped_out)
        log.info("Saved conservation + variant data")

        if not mapped_dssp_df.empty:
            mapped_data["AA"] = mapped_data.UniProt_ResNum.map(ress_AA_dict)
            mapped_data["RSA"] = mapped_data.UniProt_ResNum.map(ress_RSA_dict)
            mapped_data["SS"] = mapped_data.UniProt_ResNum.map(ress_SS_dict)
        else:
            log.warning("No DSSP data available")
            pass

        mapped_data["binding_sites"] = mapped_data.UniProt_ResNum.map(bs_ress_membership_dict)
        mapped_data.to_pickle(final_table_out)
        log.info("Saved final table")
    else:
        log.info("Not running variants")

    log.info("Variants section completed")

    ### generate binding site summary table

    bss_table_out = os.path.join(results_dir, "{}_bss_table.pkl".format(input_id))

    if OVERRIDE or not os.path.isfile(bss_table_out):
        mapped_data["job_id"] = input_id
        _, bss_data = get_bss_table(mapped_data)
        bss_data = bss_data.fillna("NaN") # pre-processing could also be done before saving the pickle
        bss_data.columns = headings # changing table column names
        bss_data["ID"] = bss_data["ID"].astype(int) # converting ID to int
        bss_data = bss_data.sort_values(by = ["ID"]).reset_index(drop = True) # sorting by ID
        bss_data.to_pickle(bss_table_out)
        log.info("Saved binding site summary table")
    else:
        pass

    log.info("THE END")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LIGYSIS: a ligand binding site analysis pipeline that clusters ligands, defines, and characterises binding sites.")
    parser.add_argument("input_dir", type=str, help="Path to directory containing input structures")
    parser.add_argument("uniprot_id", type=str, help="UniProt ID of the protein")
    parser.add_argument("struc_fmt", type=str, choices=["pdb", "mmcif"], default="mmcif", help="Format of the input structures (must be 'pdb' or 'mmcif')")
    parser.add_argument("--override", help="Override any previously generated files.", action="store_true")
    parser.add_argument("--override_variants", help="Override any previously generated files (ONLY VARIANTS SECTION).", action="store_true")
    parser.add_argument("--variants", help="Retrieves Human variants from MSA and generates tables.", action="store_true")
    parser.add_argument("--clust_method", type=str, default="average", help="Ligand clustering method (default: average)")
    parser.add_argument("--clust_dist", type=float, default=0.50, help="Ligand clustering distance threshold (default: 0.50)")
    parser.add_argument("--hmm_iters", type=int, default=3, help="Number of iterations for JACKHMMER (default: 3)")
    parser.add_argument("--cons_thresh_high", type=int, default=75, help="Conservation high threshold (default: 75)")
    parser.add_argument("--cons_thresh_low", type=int, default=25, help="Conservation low threshold (default: 25)")
    parser.add_argument("--mes_thresh", type=float, default=1.0, help="MES threshold (default: 1.0)")
    
    args = parser.parse_args()

    main(args)