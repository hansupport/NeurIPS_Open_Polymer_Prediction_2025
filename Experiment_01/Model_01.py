import os
import gc
import sys
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile


# ✅ 모든 print 출력을 로그 파일로 저장하기 위한 로거 클래스 (기존과 동일)
class TeeLogger:
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout = sys.stdout  # 원래의 stdout을 저장
        sys.stdout = self          # 현재 stdout을 이 객체로 변경

    def close(self):
        """
        로거를 종료하고 stdout을 원래대로 복원합니다.
        """
        if self.stdout:
            sys.stdout = self.stdout # 저장해두었던 원래 stdout으로 복원
            self.stdout = None # 중복 복원을 막기 위해 None으로 설정
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()

    def write(self, data):
        if self.file:
            self.file.write(data)
        if self.stdout:
            self.stdout.write(data)

    def flush(self):
        if self.file:
            self.file.flush()
        if self.stdout:
            self.stdout.flush()


# ✅ 실험 로직을 함수로 분리
def run_single_experiment(params, output_dir):
    """
    단일 실험을 실행하고 결과를 반환하는 함수
    """
    start_time = time.time()

    # Feature Engineering Flags
    DO_RDKit = params['DO_RDKit']
    DO_POLYBERT = params['DO_POLYBERT']
    DO_FINGERPRINT = params['DO_FINGERPRINT']
    RDKit_FILTER_MODE = params['RDKit_FILTER_MODE']

    # Augmentation Flags
    DO_AUGMENT_SMILES = params['DO_AUGMENT_SMILES']
    AUG_NUM = params['AUG_NUM']
    DO_GMM_AUGMENT = params['DO_GMM_AUGMENT']
    GMM_SAMPLES = params['GMM_SAMPLES']
    GMM_COMPONENTS = params['GMM_COMPONENTS']
    GMM_RANDOM_STATE = params['GMM_RANDOM_STATE']

    # Preprocessing Flags
    DO_VARIANCE_THRESHOLD = params['DO_VARIANCE_THRESHOLD']
    VARIANCE_THRESHOLD = params['VARIANCE_THRESHOLD']
    DO_StandardScaler = params['DO_StandardScaler']

    # Label & Model Flags
    N_SPLITS = params['N_SPLITS']

    # 실험 시작 로그
    print("=" * 50)
    print(f"Running experiment with settings:\n{params}")
    print("=" * 50)

    # =================================================================

    # 1. 라이브러리 임포트
    import os
    import gc
    import sys
    import joblib
    import shutil
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from collections import defaultdict
    from tqdm import tqdm

    # 모델 관련
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    # 경고 메시지 무시
    import warnings
    warnings.filterwarnings('ignore')
    torch.autograd.set_detect_anomaly(True)

    # cpu gpu 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------
    # ⚙️ 사용자 설정 영역
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    #
    # 전처리 실행 플래그
    # True로 설정 시, main() 함수를 실행하여 데이터를 새로 전처리합니다.
    # False로 설정 시, 전처리를 건너뛰고 INPUT_DIR에 지정된 기존 데이터를 사용합니다.
    RUN_PREPROCESSING = True

    # 모델 실행 플래그
    # True로 설정 시, 모델 훈련 및 검증 스크립트를 실행합니다.
    # False로 설정 시, 모델 훈련 과정을 건너뛰고 MODEL_INPUT_DIR에 지정된 기존 데이터를 사용합니다.
    RUN_MODEL_TRAINING = True

    # 제출 모드 플래그
    # True로 설정 시, 훈련과 함께, 저장된 모델로 submission.csv를 생성합니다.
    # False로 설정 시 제출
    SUBMISSION_MODE = False
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Augmentation Flags 2
    # DO_GMM_AUGMENT = False  # GMM을 이용한 데이터 증강
    # GMM_SAMPLES = 500  # GMM으로 생성할 샘플 수
    # GMM_COMPONENTS = 5  # GMM 컴포넌트 수
    # GMM_RANDOM_STATE = 42  # GMM 시드
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Normalization Flags
    # DO_VARIANCE_THRESHOLD = False  # Variance Threshold 실행 여부
    # VARIANCE_THRESHOLD = 0.1  # 최소 분산 임계값

    # DO_StandardScaler = False  # StandardScaler 실행 여부
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 전처리 실행 플래그가 True일 때, 전처리 실행할 타겟
    PREPROCESS_TARGET = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    # ------------------------------------------------------------

    # 2. 경로 설정
    # ------------------------------------------------------------
    # 2-1. input
    # ------------------------------------------------------------
    # 이전 노트북에서 생성된 파일들이 있는 디렉토리 경로를 지정하세요.
    # 참고: RUN_PREPROCESSING = False일 때 사용할 기본 경로입니다.
    #        True일 경우, 이 경로는 아래 블록에서 동적으로 변경됩니다.
    PRE_INPUT_DIR = Path("Dataset/PreprocessingData")
    # ------------------------------------------------------------
    TEST_INPUT_DIR = OUTPUT_DIR / Path("features")
    SAMPLE_SUBMISSION = Path("Dataset/neurips-open-polymer-prediction-2025")

    # 추론만 할 경우
    MODEL_INPUT_DIR = Path("Dataset/ModelData")
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 2-2. output
    # ------------------------------------------------------------
    SUBMISSION_SAVE_DIR = OUTPUT_DIR
    SUBMISSION_SAVE_DIR.mkdir(exist_ok=True, parents=True)

    MODEL_SAVE_DIR = OUTPUT_DIR / Path("models")
    MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)

    GRAPH_SAVE_DIR = OUTPUT_DIR / Path("graphs")
    GRAPH_SAVE_DIR.mkdir(exist_ok=True, parents=True)

    FOLD_GRAPH_SAVE_DIR = OUTPUT_DIR / Path("graphs/folds")
    FOLD_GRAPH_SAVE_DIR.mkdir(exist_ok=True, parents=True)

    OOF_SAVE_DIR = OUTPUT_DIR / Path("oof")
    OOF_SAVE_DIR.mkdir(exist_ok=True, parents=True)

    IMPORTANCE_SAVE_DIR = OUTPUT_DIR / Path("importance")
    IMPORTANCE_SAVE_DIR.mkdir(exist_ok=True, parents=True)

    FTT_DETAIL_SAVE_DIR = OUTPUT_DIR / Path("FTTdetails")
    FTT_DETAIL_SAVE_DIR.mkdir(exist_ok=True, parents=True)
    # ------------------------------------------------------------

    # 3. K-Fold, FTT, ET 설정
    # ------------------------------------------------------------
    # K-Fold 설정
    # N_SPLITS = 2
    RANDOM_STATE = 42

    # FTT 설정
    CATEGORICAL_THRESHOLD = 20

    # 'LGBM', 'XGB', 'CAT', 'ET' 시각화 스텝 설정
    STEP = 10
    # ------------------------------------------------------------

    # 5. 모델 선택
    # 각 타겟에 어떤 모델을 사용할지 지정합니다.
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    # 사용 가능한 모델: 'LGBM', 'XGB', 'CAT', 'ET', 'FTT'
    # ------> 모델 여러개 또는 0개 선택 가능 [] or ['LGBM'] or ['LGBM', 'XGB', 'CAT', 'ET', 'FTT']
    MODEL_CONFIG = {
        'Tg': ['LGBM', 'XGB', 'CAT', 'ET', 'FTT'],
        'FFV': ['LGBM', 'XGB', 'CAT', 'ET', 'FTT'],
        'Tc': ['LGBM', 'XGB', 'CAT', 'ET', 'FTT'],
        'Density': ['LGBM', 'XGB', 'CAT', 'ET', 'FTT'],
        'Rg': ['LGBM', 'XGB', 'CAT', 'ET', 'FTT'],
    }
    # ------------------------------------------------------------

    # 6. 하이퍼파라미터 설정
    # 각 모델의 하이퍼파라미터를 이곳에서 조정하세요.
    HPARAMS = {
        'LGBM': {
            'objective': 'mae',
            'metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'seed': RANDOM_STATE,
        },
        'XGB': {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'tree_method': 'hist',  # GPU 사용 시 'gpu_hist'
        },
        'CAT': {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': RANDOM_STATE,
            'verbose': 200,
            'allow_writing_files': False,
        },
        'ET': {  # Extra Trees
            'n_estimators': 500,
            'criterion': 'squared_error',  # MAE는 지원되지 않음
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
        },
        'FTT': {
            # 1. 모델 아키텍처 파라미터
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 3,
            'ff_hidden_dim': 128,
            'dropout': 0.1,

            # 2. 모델 학습 파라미터
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'early_stopping_patience': 15,
        },
    }

    # ------------------------------------------------------------
    #  1. 사용자 설정
    # ------------------------------------------------------------
    # Feature Extraction Flags
    # DO_RDKit = True  # RDKit 특성추출      ---┯--> 둘 중 하나 또는 둘 다 사용
    # DO_POLYBERT = False  # PolyBERT 임베딩    ---┛
    # DO_FINGERPRINT = False  # 핑거프린트(MFP, MACCS) 특성 추가
    # RDKit_FILTER_MODE = 'useless'
    #    'required' → required_descriptors ∪ filters_required[label] 사용 (공통 + 라벨별 필요만 포함)
    #    'useless'  → useless_cols ∪ filters_useless[label] 드롭 (공통 + 라벨별 불필요만 제거)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Augmentation Flags 1
    # DO_AUGMENT_SMILES = False  # SMILES Augmentation 실행 여부
    # AUG_NUM = 1  # SMILES 당 Augmentation 수
    # ------------------------------------------------------------

    # ------------------------------------------------------------

    # 1) 필요한 특성만 추출

    # 공통
    required_descriptors = {
        'graph_diameter', 'avg_shortest_path', 'num_cycles',
        'MolWt', 'MolLogP', 'TPSA', 'RotatableBonds', 'NumAtoms'
    }
    # 라벨별
    filters_required = {
        'Tg': list(set([
            'BalabanJ', 'BertzCT', 'Chi1', 'Chi3n', 'Chi4n', 'EState_VSA4', 'EState_VSA8',
            'FpDensityMorgan3', 'HallKierAlpha', 'Kappa3', 'MaxAbsEStateIndex', 'MolLogP',
            'NumAmideBonds', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds',
            'PEOE_VSA14', 'Phi', 'RingCount', 'SMR_VSA1', 'SPS', 'SlogP_VSA1', 'SlogP_VSA5',
            'SlogP_VSA8', 'TPSA', 'VSA_EState1', 'VSA_EState4', 'VSA_EState6', 'VSA_EState7',
            'VSA_EState8', 'fr_C_O_noCOO', 'fr_NH1', 'fr_benzene', 'fr_bicyclic', 'fr_ether',
            'fr_unbrch_alkane'
        ]).union(required_descriptors)),

        'FFV': list(set([
            'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
            'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'EState_VSA10', 'EState_VSA5',
            'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1',
            'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha',
            'HeavyAtomMolWt', 'Kappa1', 'Kappa2', 'Kappa3', 'MaxAbsEStateIndex',
            'MaxEStateIndex', 'MinEStateIndex', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount',
            'NOCount', 'NumAromaticHeterocycles', 'NumHAcceptors', 'NumHDonors',
            'NumHeterocycles', 'NumRotatableBonds', 'PEOE_VSA14', 'RingCount', 'SMR_VSA1',
            'SMR_VSA10', 'SMR_VSA3', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SPS',
            'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
            'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
            'SlogP_VSA8', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
            'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
            'VSA_EState8', 'VSA_EState9', 'fr_Ar_N', 'fr_C_O', 'fr_NH0', 'fr_NH1',
            'fr_aniline', 'fr_ether', 'fr_halogen', 'fr_thiophene'
        ]).union(required_descriptors)),

        'Tc': list(set([
            'BalabanJ', 'BertzCT', 'Chi0', 'EState_VSA5', 'ExactMolWt', 'FpDensityMorgan1',
            'FpDensityMorgan2', 'FpDensityMorgan3', 'HeavyAtomMolWt', 'MinEStateIndex',
            'MolWt', 'NumAtomStereoCenters', 'NumRotatableBonds', 'NumValenceElectrons',
            'SMR_VSA10', 'SMR_VSA7', 'SPS', 'SlogP_VSA6', 'SlogP_VSA8', 'VSA_EState1',
            'VSA_EState7', 'fr_NH1', 'fr_ester', 'fr_halogen'
        ]).union(required_descriptors)),

        'Density': list(set([
            'BalabanJ', 'Chi3n', 'Chi3v', 'Chi4n', 'EState_VSA1', 'ExactMolWt',
            'FractionCSP3', 'HallKierAlpha', 'Kappa2', 'MinEStateIndex', 'MolMR', 'MolWt',
            'NumAliphaticCarbocycles', 'NumHAcceptors', 'NumHeteroatoms',
            'NumRotatableBonds', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA12', 'SlogP_VSA5',
            'TPSA', 'VSA_EState10', 'VSA_EState7', 'VSA_EState8'
        ]).union(required_descriptors)),

        'Rg': list(set([
            'AvgIpc', 'Chi0n', 'Chi1v', 'Chi2n', 'Chi3v', 'ExactMolWt', 'FpDensityMorgan1',
            'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha', 'HeavyAtomMolWt',
            'Kappa3', 'MaxAbsEStateIndex', 'MolWt', 'NOCount', 'NumRotatableBonds',
            'NumUnspecifiedAtomStereoCenters', 'NumValenceElectrons', 'PEOE_VSA14',
            'PEOE_VSA6', 'SMR_VSA1', 'SMR_VSA5', 'SPS', 'SlogP_VSA1', 'SlogP_VSA2',
            'SlogP_VSA7', 'SlogP_VSA8', 'VSA_EState1', 'VSA_EState8', 'fr_alkyl_halide',
            'fr_halogen'
        ]).union(required_descriptors))
    }

    # 드롭할 불필요 컬럼 리스트
    useless_cols = [

        'MaxPartialCharge',
        # Nan data
        'BCUT2D_MWHI',
        'BCUT2D_MWLOW',
        'BCUT2D_CHGHI',
        'BCUT2D_CHGLO',
        'BCUT2D_LOGPHI',
        'BCUT2D_LOGPLOW',
        'BCUT2D_MRHI',
        'BCUT2D_MRLOW',

        # Constant data
        'NumRadicalElectrons',
        'SMR_VSA8',
        'SlogP_VSA9',
        'fr_barbitur',
        'fr_benzodiazepine',
        'fr_dihydropyridine',
        'fr_epoxide',
        'fr_isothiocyan',
        'fr_lactam',
        'fr_nitroso',
        'fr_prisulfonamd',
        'fr_thiocyan',

        # High correlated data >0.95
        'MaxEStateIndex',
        'HeavyAtomMolWt',
        'ExactMolWt',
        'NumValenceElectrons',
        'Chi0',
        'Chi0n',
        'Chi0v',
        'Chi1',
        'Chi1n',
        'Chi1v',
        'Chi2n',
        'Kappa1',
        'LabuteASA',
        'HeavyAtomCount',
        'MolMR',
        'Chi3n',
        'BertzCT',
        'Chi2v',
        'Chi4n',
        'HallKierAlpha',
        'Chi3v',
        'Chi4v',
        'MinAbsPartialCharge',
        'MinPartialCharge',
        'MaxAbsPartialCharge',
        'FpDensityMorgan2',
        'FpDensityMorgan3',
        'Phi',
        'Kappa3',
        'fr_nitrile',
        'SlogP_VSA6',
        'NumAromaticCarbocycles',
        'NumAromaticRings',
        'fr_benzene',
        'VSA_EState6',
        'NOCount',
        'fr_C_O',
        'fr_C_O_noCOO',
        'NumHDonors',
        'fr_amide',
        'fr_Nhpyrrole',
        'fr_phenol',
        'fr_phenol_noOrthoHbond',
        'fr_COO2',
        'fr_halogen',
        'fr_diazo',
        'fr_nitro_arom',
        'fr_phos_ester'
    ]

    filters_useless = {
        'Tg': [],
        'FFV': [],
        'Tc': [],
        'Density': [],
        'Rg': [],
    }

    # 2. 라이브러리 로드
    import os
    import re
    from pathlib import Path
    import joblib
    import math

    import numpy as np
    import pandas as pd

    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.mixture import GaussianMixture

    import torch
    from sentence_transformers.SentenceTransformer import SentenceTransformer

    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    import networkx as nx
    from tqdm import tqdm

    # 3. 경로 설정
    INPUT_DIR = Path("Dataset/neurips-open-polymer-prediction-2025")
    TRAIN_CSV = INPUT_DIR / "train.csv"
    TEST_CSV = INPUT_DIR / "test.csv"
    WORK_DIR = OUTPUT_DIR / Path("features")
    WORK_DIR.mkdir(exist_ok=True, parents=True)

    # PolyBERT 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = Path("Dataset") / "polyBERT" / "polyBERT-local"
    polybert = SentenceTransformer(str(MODEL_PATH.resolve()), device=device)

    # 4. 유틸 함수
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    def get_canonical_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None

    def clean_and_validate_smiles(smiles):
        bad_patterns = [
            '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
            "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
            '([R])', '([R1])', '([R2])'
        ]
        if not isinstance(smiles, str) or not smiles:
            return None
        for p in bad_patterns:
            if p in smiles:
                return None
        if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
            return None
        return get_canonical_smiles(smiles)

    def separate_subtables(train_df):
        return {
            label: train_df.loc[train_df[label].notna(), ['SMILES', label]].reset_index(drop=True)
            for label in TARGETS
        }

    def augment_smiles_dataset(smiles_list, labels, num_augments=AUG_NUM):
        aug_smi, aug_lbl = [], []
        for smi, lab in zip(smiles_list, labels):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            aug_smi.append(smi);
            aug_lbl.append(lab)
            for _ in range(num_augments):
                aug_smi.append(Chem.MolToSmiles(mol, doRandom=True))
                aug_lbl.append(lab)
        return aug_smi, np.array(aug_lbl)

    def smiles_to_combined_fingerprints_with_descriptors(smiles_list, radius=2, n_bits=128):
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fps, descs, valid, invalid_idx = [], [], [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mfp = np.array(gen.GetFingerprint(mol))
                maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
                fps.append(np.concatenate([mfp, maccs]))
                dv = {}
                for name, fn in Descriptors.descList:
                    try:
                        dv[name] = fn(mol)
                    except:
                        dv[name] = np.nan
                dv.update({
                    'MolWt': Chem.Descriptors.MolWt(mol),
                    'LogP': Chem.Descriptors.MolLogP(mol),
                    'TPSA': rdMolDescriptors.CalcTPSA(mol),
                    'RotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'NumAtoms': mol.GetNumAtoms()
                })
                adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)
                if nx.is_connected(G):
                    dv['graph_diameter'] = nx.diameter(G)
                    dv['avg_shortest_path'] = nx.average_shortest_path_length(G)
                else:
                    dv['graph_diameter'] = dv['avg_shortest_path'] = 0
                dv['num_cycles'] = len(list(nx.cycle_basis(G)))
                descs.append(dv);
                valid.append(smi)
            else:
                fps.append(np.zeros(n_bits + 167))
                descs.append({});
                invalid_idx.append(i);
                valid.append(None)
        return np.array(fps), descs, valid, invalid_idx

    def extract_polybert(smiles_list):
        emb = polybert.encode(
            smiles_list,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return normalize(emb, axis=1)

    print("Library loading complete")

    # 5. 외부 데이터셋 설정

    # 1. 기본 데이터 로드
    BASE_PATH = 'Dataset/neurips-open-polymer-prediction-2025/'
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    print("📂 Loading base train/test data...")
    train = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Clean SMILES 함수가 반드시 정의되어 있어야 함
    train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
    test_df['SMILES'] = test_df['SMILES'].apply(clean_and_validate_smiles)

    train = train[train['SMILES'].notnull()].reset_index(drop=True)

    print(f"✅ Base training samples: {len(train)}")
    print(f"✅ Base test samples: {len(test_df)}")

    print("\n📂 Loading external datasets...")

    # 2. 외부 데이터셋을 안전하게 로드
    external_datasets = []

    def safe_load_dataset(path, target, processor_func, description):
        try:
            if path.endswith('.xlsx'):
                data = pd.read_excel(path)
            else:
                data = pd.read_csv(path)
            data = processor_func(data)
            external_datasets.append((target, data))
            print(f"   ✅ {description}: {len(data)} samples")
            return True
        except Exception as e:
            print(f"   ⚠️ {description} failed: {str(e)[:100]}")
            return False

    # 3. 외부 데이터 로드
    safe_load_dataset('Dataset/tc_SMILES/tc_SMILES.csv', 'Tc',
                      lambda df: df.rename(columns={'TC_mean': 'Tc'}),
                      'Tc data')

    safe_load_dataset('Dataset/tg_SMILES_PID_Polymer Class/tgSS_enriched_cleaned.csv', 'Tg',
                      lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
                      'TgSS enriched data')

    safe_load_dataset('Dataset/smiles-extra-data/JCIM_sup_bigsmiles.csv', 'Tg',
                      lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
                      'JCIM Tg data')

    safe_load_dataset('Dataset/smiles-extra-data/data_dnst1.xlsx', 'Density',
                      lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                      .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                      .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
                      'Density data')

    # 4. 외부 데이터 병합
    def add_extra_data_clean(df_train, df_extra, target):
        # 외부 데이터셋에서 SMILES 정제 후, target 값이 있는 데이터만 df_train에 병합
        df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
        df_extra = df_extra[df_extra['SMILES'].notnull()]
        df_extra = df_extra.dropna(subset=[target])
        if len(df_extra) == 0:
            return df_train

        # SMILES별 평균값 (중복 SMILES 처리)
        df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

        # train에 없는 새로운 SMILES만 추가
        unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])
        extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()

        if len(extra_to_add) > 0:
            # 모든 타겟 컬럼을 맞춰서 추가
            for col in TARGETS:
                if col not in extra_to_add.columns:
                    extra_to_add[col] = np.nan
            extra_to_add = extra_to_add[['SMILES'] + TARGETS]
            df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

        return df_train

    # 5. 병합 실행
    train_extended = train[['SMILES'] + TARGETS].copy()
    for target, dataset in external_datasets:
        train_extended = add_extra_data_clean(train_extended, dataset, target)

    # 6. 합 후 최종 clean 처리
    train_extended['SMILES'] = train_extended['SMILES'].apply(clean_and_validate_smiles)
    train_extended = train_extended[train_extended['SMILES'].notnull()].reset_index(drop=True)
    train_df = train_extended

    print(f"\n✅ Final extended training samples: {len(train_extended)}")

    def main(target_labels=None):
        """
        target_labels: list of labels to process (e.g. ['Tg','Density'])
                       None 이면 모든 TARGETS 처리
        """
        labels = TARGETS if target_labels is None else target_labels

        # subtables 준비
        subtables = {
            lbl: train_extended.loc[train_extended[lbl].notna(), ['SMILES', lbl]]
            .reset_index(drop=True)
            for lbl in labels
        }

        for label in labels:
            # ▶ drop 리스트 정의 (루프 안에서)
            drop_useless = list(set(useless_cols) | set(filters_useless[label]))
            drop_required = [
                'BCUT2D_MWLOW', 'BCUT2D_MWHI', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
                'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRLOW', 'BCUT2D_MRHI',
                'MinAbsPartialCharge', 'MaxPartialCharge',
                'MinPartialCharge', 'MaxAbsPartialCharge', 'SMILES'
            ]

            df_lbl = subtables[label]
            smiles = df_lbl['SMILES'].tolist()
            y_tr = df_lbl[label].values

            # 1) SMILES 증강
            if DO_AUGMENT_SMILES:
                print(f"\n[{label}] 1) SMILES 증강")
                aug_smiles, aug_labels = [], []
                for smi, lab in tqdm(zip(smiles, y_tr),
                                     total=len(smiles),
                                     desc=f"{label} SMILES 증강"):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    aug_smiles.append(smi)
                    aug_labels.append(lab)
                    for _ in range(AUG_NUM):
                        aug_smiles.append(Chem.MolToSmiles(mol, doRandom=True))
                        aug_labels.append(lab)
                smiles, y_tr = aug_smiles, np.array(aug_labels)
            else:
                print(f"[{label}] 1) SMILES 증강 skipped (DO_AUGMENT_SMILES=False)")

            # 2) RDKit descriptor 및 핑거프린트 추출
            if DO_RDKit:
                print(f"[{label}] 2) RDKit descriptor 추출 & 필터링")
                fps, descs = [], []
                for smi in tqdm(smiles,
                                total=len(smiles),
                                desc=f"{label} RDKit 추출"):
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        if DO_FINGERPRINT:
                            mfp = np.array(GetMorganGenerator(radius=2, fpSize=128).GetFingerprint(mol))
                            maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
                            fps.append(np.concatenate([mfp, maccs]))
                        dv = {}
                        for name, fn in Descriptors.descList:
                            try:
                                dv[name] = fn(mol)
                            except:
                                dv[name] = np.nan
                        dv.update({
                            'MolWt': Chem.Descriptors.MolWt(mol),
                            'LogP': Chem.Descriptors.MolLogP(mol),
                            'TPSA': rdMolDescriptors.CalcTPSA(mol),
                            'RotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                            'NumAtoms': mol.GetNumAtoms()
                        })
                        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
                        G = nx.from_numpy_array(adj)
                        if nx.is_connected(G):
                            dv['graph_diameter'] = nx.diameter(G)
                            dv['avg_shortest_path'] = nx.average_shortest_path_length(G)
                        else:
                            dv['graph_diameter'] = dv['avg_shortest_path'] = 0
                        dv['num_cycles'] = len(list(nx.cycle_basis(G)))
                        descs.append(dv)
                    else:
                        if DO_FINGERPRINT:
                            fps.append(np.zeros(128 + 167))
                        descs.append({})

                # RDKit 서술자 DataFrame 생성 및 필터링
                df_desc = pd.DataFrame(descs).fillna(0)
                if RDKit_FILTER_MODE == 'useless':
                    # 'useless' 모드: 불필요한 컬럼들을 정의하고 제거합니다.
                    cols_to_drop = list(set(useless_cols) | set(filters_useless.get(label, set())))
                    df_desc.drop(columns=cols_to_drop, errors='ignore', inplace=True)
                    print(f"[{label}] 'useless' 모드 적용. {len(cols_to_drop)}개 규칙으로 특성 제거.")

                    # 필터링 후, inf 값을 NaN으로 변경하고, NaN을 0으로 채우기
                    df_desc.replace([np.inf, -np.inf], np.nan, inplace=True)

                elif RDKit_FILTER_MODE == 'required':
                    # 'required' 모드: 필요한 컬럼들만 선택합니다.
                    # filters_required[label]은 이미 공통+라벨별 리스트의 합집합으로 정의되어 있습니다.
                    required_cols = filters_required.get(label, [])
                    df_desc = df_desc.filter(items=required_cols, axis=1)
                    print(f"[{label}] 'required' 모드 적용. {len(required_cols)}개 특성 선택.")

                # 핑거프린트 DataFrame 생성
                if DO_FINGERPRINT:
                    fp_cols = [f'MFP_{i}' for i in range(128)] + [f'MACCS_{i}' for i in range(167)]
                    df_fps = pd.DataFrame(fps, columns=fp_cols)
                else:
                    df_fps = pd.DataFrame(index=range(len(smiles)))
            else:
                print(f"[{label}] 2) RDKit descriptor 추출 skipped (DO_RDKit=False)")
                # 빈 DataFrame 할당
                df_desc = pd.DataFrame(index=range(len(smiles)))
                df_fps = pd.DataFrame(index=range(len(smiles)))

            # 3) PolyBERT embedding
            if DO_POLYBERT:
                print(f"[{label}] 3) PolyBERT embedding")
                emb_list = []
                batch_size = 64
                for i in tqdm(range(0, len(smiles), batch_size),
                              total=math.ceil(len(smiles) / batch_size),
                              desc=f"{label} PolyBERT embedding"):
                    batch = smiles[i: i + batch_size]
                    emb_batch = polybert.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                    emb_list.append(emb_batch)
                emb = normalize(np.vstack(emb_list), axis=1)
            else:
                print(f"[{label}] 3) PolyBERT embedding skipped (DO_POLYBERT=False)")
                emb = np.empty((len(smiles), 0), dtype=np.float64)

            # 4) Feature 결합 (DataFrame 사용)
            print(f"[{label}] 4) Feature 결합")
            # RDKit 또는 PolyBERT 중 하나라도 켜져 있어야 합니다.
            if not (DO_RDKit or DO_POLYBERT):
                raise RuntimeError(
                    f"[{label}] No features to combine: both DO_RDKit and DO_POLYBERT are False."
                )

            # PolyBERT 임베딩을 DataFrame으로 만듭니다.
            df_emb = pd.DataFrame(emb, columns=[f'PolyBERT_{i}' for i in range(emb.shape[1])])

            # 모든 특성 DataFrame을 수평으로 결합합니다.
            for _ in tqdm(range(1), desc=f"{label} Feature 결합"):
                X_tr_df = pd.concat([df_desc, df_fps, df_emb], axis=1)

                # 무한대(inf) 값을 NaN으로 변환하고 0으로 채우기
                X_tr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_tr_df.fillna(0, inplace=True)

                print(f"[{label}] 결합 후 특성 개수: {X_tr_df.shape[1]}")

            # 최종 데이터와 컬럼 이름 저장
            X_tr_final_df = X_tr_df
            final_feature_names = X_tr_final_df.columns.tolist()
            joblib.dump(final_feature_names, WORK_DIR / f"all_feature_names_{label}.pkl")

            X_tr_final_np = X_tr_final_df.values

            # 데이터 저장
            np.save(WORK_DIR / f"X_train_{label}.npy", X_tr_final_np)
            np.save(WORK_DIR / f"y_train_{label}.npy", y_tr)

            print(f"✅ [{label}] 최종 데이터(shape:{X_tr_final_np.shape})와 "
                  f"{len(final_feature_names)}개의 특성 이름 저장 완료.")

        train_extended.to_csv(WORK_DIR / "train_merged.csv", index=False)
        print("✅ 모든 라벨 처리 완료.")

    def main_test(target_labels=None):
        """
        테스트 데이터 전처리 및 저장 (raw features만 생성)
        - 모델 파이프라인이 스스로 동일 전처리 규칙을 적용하므로,
          main_test 단계에선 오직 feature matrix 생성/저장만 수행
        """
        labels = TARGETS if target_labels is None else target_labels
        smis_te, n_samples = test_df['SMILES'].tolist(), len(test_df)

        for label in labels:
            print(f"\n[{label}] Test Data Preprocessing for {label}")

            # ▶ drop 리스트 정의
            drop_useless = list(set(useless_cols) | set(filters_useless[label]))
            drop_required = [
                'BCUT2D_MWLOW', 'BCUT2D_MWHI', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
                'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRLOW', 'BCUT2D_MRHI',
                'MinAbsPartialCharge', 'MaxPartialCharge',
                'MinPartialCharge', 'MaxAbsPartialCharge', 'SMILES'
            ]

            # 1) SMILES 증강 (테스트 데이터에는 적용하지 않음)
            print(f"[{label}] 1) SMILES 증강 skipped (Test data)")

            # 2) RDKit descriptor 및 fingerprint 추출
            if DO_RDKit:
                print(f"[{label}] 2) Test RDKit descriptor & fingerprint 추출")
                fps_te, descs_te = [], []
                for smi in tqdm(smis_te, total=n_samples, desc=f"{label} Test RDKit"):
                    mol = None
                    if isinstance(smi, str):
                        mol = Chem.MolFromSmiles(smi)

                    if mol:
                        if DO_FINGERPRINT:
                            mfp = np.array(GetMorganGenerator(radius=2, fpSize=128).GetFingerprint(mol))
                            maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
                            fps_te.append(np.concatenate([mfp, maccs]))
                        dv = {}
                        for name, fn in Descriptors.descList:
                            try:
                                dv[name] = fn(mol)
                            except:
                                dv[name] = np.nan
                        dv.update({
                            'MolWt': Chem.Descriptors.MolWt(mol), 'LogP': Chem.Descriptors.MolLogP(mol),
                            'TPSA': rdMolDescriptors.CalcTPSA(mol),
                            'RotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                            'NumAtoms': mol.GetNumAtoms()
                        })
                        adj = Chem.rdmolops.GetAdjacencyMatrix(mol);
                        G = nx.from_numpy_array(adj)
                        if nx.is_connected(G):
                            dv['graph_diameter'] = nx.diameter(G);
                            dv['avg_shortest_path'] = nx.average_shortest_path_length(G)
                        else:
                            dv['graph_diameter'] = dv['avg_shortest_path'] = 0
                        dv['num_cycles'] = len(list(nx.cycle_basis(G)))
                        descs_te.append(dv)
                    else:
                        if DO_FINGERPRINT: fps_te.append(np.zeros(128 + 167))
                        descs_te.append({})
                df_desc_te = pd.DataFrame(descs_te).fillna(0)

                if RDKit_FILTER_MODE == 'useless':
                    # 공통 useless_cols + 라벨별 filters_useless[label] 을 drop
                    cols_to_drop = list(set(useless_cols) | set(filters_useless[label]))
                    df_desc_te.drop(columns=cols_to_drop, errors='ignore', inplace=True)
                elif RDKit_FILTER_MODE == 'required':
                    # 필요한 컬럼만 남김
                    req = filters_required[label]  # 이미 required_descriptors ∪ 라벨별 필요 리스트로 구성돼 있음
                    df_desc_te = df_desc_te[req]

                if DO_FINGERPRINT:
                    fp_cols = [f'MFP_{i}' for i in range(128)] + [f'MACCS_{i}' for i in range(167)]
                    df_fps_te = pd.DataFrame(fps_te, columns=fp_cols)
                else:
                    df_fps_te = pd.DataFrame(index=range(n_samples))
            else:
                print(f"[{label}] 2) Test RDKit descriptor 추출 skipped (DO_RDKit=False)")
                df_desc_te = pd.DataFrame(index=range(n_samples))
                df_fps_te = pd.DataFrame(index=range(n_samples))

            # 3) PolyBERT embedding
            if DO_POLYBERT:
                print(f"[{label}] 3) Test PolyBERT embedding")
                emb_te_list = []
                batch_size = 64
                for i in tqdm(range(0, n_samples, batch_size),
                              total=math.ceil(n_samples / batch_size),
                              desc=f"{label} Test PolyBERT embedding"):
                    batch = smis_te[i: i + batch_size]

                    safe_batch = [s if isinstance(s, str) else '' for s in batch]
                    emb_batch = polybert.encode(safe_batch, convert_to_numpy=True, show_progress_bar=False)
                    emb_te_list.append(emb_batch)

                emb_te = normalize(np.vstack(emb_te_list), axis=1)
            else:
                print(f"[{label}] 3) PolyBERT embedding skipped (DO_POLYBERT=False)")
                emb_te = np.empty((n_samples, 0), dtype=np.float64)

            # 4) Feature 결합
            print(f"[{label}] 4) Feature 결합")
            if not (DO_RDKit or DO_POLYBERT):
                raise RuntimeError(f"[{label}] No features to combine: both DO_RDKit and DO_POLYBERT are False.")
            df_emb_te = pd.DataFrame(emb_te, columns=[f'PolyBERT_{i}' for i in range(emb_te.shape[1])])
            X_te_df_initial = pd.concat([df_desc_te, df_fps_te, df_emb_te], axis=1)

            # 무한대(inf) 값을 NaN으로 변환
            X_te_df_initial.replace([np.inf, -np.inf], np.nan, inplace=True)

            print(f"[{label}] 결합 후 특성 개수: {X_te_df_initial.shape[1]}")

            # 훈련시 저장된 전체 feature 목록 불러오기, 순서 맞추기만 수행
            all_feature_names = joblib.load(INPUT_DIR / f"all_feature_names_{label}.pkl")
            X_te_df_initial = X_te_df_initial.reindex(columns=all_feature_names, fill_value=0)
            print(f"[{label}] Test 데이터 특성을 훈련 스키마에 맞춰 재정렬 완료. Shape: {X_te_df_initial.shape}")

            # 6) npy 저장
            X_te_processed = X_te_df_initial.values
            np.save(WORK_DIR / f"X_test_{label}.npy", X_te_processed)
            print(f"✅ [{label}] Final test data saved. Shape: {X_te_processed.shape}")

        test_df.to_csv(WORK_DIR / "test_cleaned.csv", index=False)
        print("✅ All test features processed.")

    # ------------------------------------------------------------
    # ⚡️ (선택) 데이터 전처리 실행
    # ------------------------------------------------------------
    # RUN_PREPROCESSING 플래그 값에 따라 데이터 전처리를 실행합니다.

    if RUN_PREPROCESSING:
        print("RUN_PREPROCESSING=True. 데이터 전처리를 시작합니다...")

        # 전처리, 전체 TARGETS 처리: main() 또는 일부만: main(['Tg','Density'])
        main(PREPROCESS_TARGET)

        # 전처리가 완료되었으므로, 이후 훈련 단계에서 사용할 데이터 경로를
        # 전처리 결과물이 있는 폴더로 변경합니다.
        INPUT_DIR = OUTPUT_DIR / Path("features")
        print(f"✅ 전처리 완료. 모델 훈련을 위한 INPUT_DIR이 '{INPUT_DIR}'(으)로 변경되었습니다.")
    else:
        INPUT_DIR = PRE_INPUT_DIR
        print(f"Skipped Preprocessing (RUN_PREPROCESSING=False)")

    # ------------------------------------------------------------
    # 🧠 FTTransformer 모델 클래스
    # 📦 PyTorch 데이터셋 및 헬퍼 함수
    # ------------------------------------------------------------
    class FTEmbedding(nn.Module):
        def __init__(self, categories, num_continuous, embedding_dim):
            super().__init__()
            # 범주형 특성이 있을 때만 임베딩 레이어를 생성
            if categories:
                self.cat_embeddings = nn.ModuleList([
                    nn.Embedding(num_cat, embedding_dim) for num_cat in categories
                ])
            else:
                self.cat_embeddings = None

            self.cont_emb = nn.Linear(num_continuous, embedding_dim) if num_continuous > 0 else None
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

            if self.cat_embeddings:
                for emb in self.cat_embeddings:
                    nn.init.xavier_uniform_(emb.weight)
            if self.cont_emb:
                nn.init.xavier_uniform_(self.cont_emb.weight)

        def forward(self, x_cat, x_cont):
            B = x_cat.size(0) if x_cat.nelement() > 0 else x_cont.size(0)
            tokens_list = []

            # 범주형 특성이 있을 때만 임베딩 및 스택 연산 수행
            if self.cat_embeddings and x_cat.nelement() > 0:
                cat_tokens = torch.stack([
                    emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)
                ], dim=1)
                tokens_list.append(cat_tokens)

            if x_cont is not None and self.cont_emb and x_cont.nelement() > 0:
                cont_token = self.cont_emb(x_cont).unsqueeze(1)
                tokens_list.append(cont_token)

            # 리스트 맨 앞에 CLS 토큰 추가
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens_list.insert(0, cls_tokens)

            if not tokens_list:
                # 만약 범주형, 연속형 특성이 모두 없는 극단적인 경우
                return self.cls_token.expand(B, -1, -1)

            return torch.cat(tokens_list, dim=1)

    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(ff_hidden_dim, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))
            return x

    class FTTransformer(nn.Module):
        def __init__(self, categories, num_continuous, embedding_dim, num_heads,
                     num_layers, ff_hidden_dim, dropout=0.1, output_dim=1):
            super().__init__()
            self.embedding = FTEmbedding(categories, num_continuous, embedding_dim)
            self.transformer_blocks = nn.Sequential(*[
                TransformerBlock(embedding_dim, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ])
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, ff_hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(ff_hidden_dim, output_dim)
            )

        def forward(self, x_cat, x_cont):
            tokens = self.embedding(x_cat, x_cont)
            tokens = self.transformer_blocks(tokens)
            cls_token = tokens[:, 0]
            return self.mlp_head(cls_token).squeeze(-1)

    class TabularDataset(Dataset):
        def __init__(self, X_cat, X_cont, y=None):
            self.X_cat = torch.tensor(X_cat, dtype=torch.long)
            self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

            self.length = len(self.X_cat) if self.X_cat.nelement() > 0 else len(self.X_cont)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            if self.y is not None:
                return self.X_cat[idx], self.X_cont[idx], self.y[idx]
            return self.X_cat[idx], self.X_cont[idx]

    def get_model(model_name, hparams, categories=None, num_continuous=None):
        if model_name == 'LGBM': return lgb.LGBMRegressor(**hparams)
        if model_name == 'XGB': return xgb.XGBRegressor(**hparams)
        if model_name == 'CAT': return cb.CatBoostRegressor(**hparams)
        if model_name == 'ET': return ExtraTreesRegressor(**hparams)
        if model_name == 'FTT':
            return FTTransformer(
                categories=categories, num_continuous=num_continuous,
                embedding_dim=hparams['embedding_dim'], num_heads=hparams['num_heads'],
                num_layers=hparams['num_layers'], ff_hidden_dim=hparams['ff_hidden_dim'],
                dropout=hparams['dropout']
            ).to(DEVICE)
        raise ValueError(f"Unknown model: {model_name}")

    def get_tree_model_history(model, model_name):
        """LGBM, XGB, CAT 모델에서 훈련 및 검증 기록을 모두 추출합니다."""
        try:
            if model_name == 'LGBM':
                results = model.evals_result_
                # 해결책: Metric 이름을 하드코딩하는 대신, 동적으로 첫 번째 키를 가져옵니다.
                #   이렇게 하면 LightGBM이 'mae'를 쓰든 'l1'을 쓰든 상관없이 작동합니다.
                train_metric_key = list(results['train'].keys())[0]
                valid_metric_key = list(results['valid'].keys())[0]

                return {
                    'train_metric': results['train'][train_metric_key],
                    'val_metric': results['valid'][valid_metric_key],
                    'metric_name': 'MAE'  # 그래프에 표시될 이름은 'MAE'로 고정
                }
            if model_name == 'XGB':
                results = model.evals_result()
                return {
                    'train_metric': results['validation_0']['mae'],
                    'val_metric': results['validation_1']['mae'],
                    'metric_name': 'MAE'
                }
            if model_name == 'CAT':
                results = model.get_evals_result()
                return {
                    'train_metric': results['learn']['MAE'],
                    'val_metric': results['validation_1']['MAE'],
                    'metric_name': 'MAE'
                }
        except (AttributeError, KeyError, IndexError) as e:
            # 오류 발생 시, 어떤 오류인지와 함께 디버깅 정보를 상세히 출력합니다.
            print(f"DEBUG: Could not get history for {model_name}. Error: {e}")

            # 추가 디버깅 정보: 사용 가능한 키 목록을 출력 시도
            if hasattr(model, 'evals_result_'):  # LightGBM
                print(f"DEBUG: Available keys in evals_result_: {model.evals_result_.keys()}")
            elif hasattr(model, 'get_evals_result'):  # CatBoost
                print(f"DEBUG: Available keys in get_evals_result: {model.get_evals_result().keys()}")

            return None

    class FTTWrapper:
        """
        FT-Transformer 클래스입니다.
        """

        def __init__(self, model_params, device='cuda'):
            self.model = None
            self.model_params = model_params
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.preprocessor_ = {}
            self.history_ = {}
            self.selector_ = None

        def fit(self, X_train, y_train, X_val, y_val, feature_names, cat_threshold, best_model_save_path,
                do_variance_threshold, variance_threshold_val, do_standard_scaler):
            """
            분할된 훈련/검증 데이터를 받아 전처리, 모델 훈련, 최고 모델 저장을 수행합니다.
            """
            if do_variance_threshold:
                self.selector_ = VarianceThreshold(threshold=variance_threshold_val)
                X_train = self.selector_.fit_transform(X_train)
                X_val = self.selector_.transform(X_val)
                # 특성 선택 후, feature_names 리스트도 업데이트합니다.
                feature_names = [name for name, keep in zip(feature_names, self.selector_.get_support()) if keep]

            # --- 1. 데이터 전처리 (훈련 데이터 기준으로 fit) ---
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            cat_cols = [c for c in X_train_df.columns if X_train_df[c].astype(str).nunique() <= cat_threshold]
            cont_cols = [c for c in X_train_df.columns if c not in cat_cols]

            encoders = {c: LabelEncoder().fit(X_train_df[c].astype(str)) for c in cat_cols}
            for c, enc in encoders.items():
                X_train_df[c] = enc.transform(X_train_df[c].astype(str))

            scaler = None
            if do_standard_scaler and cont_cols:
                scaler = StandardScaler()
                X_train_df[cont_cols] = scaler.fit_transform(X_train_df[cont_cols])

            # 훈련된 전처리기를 클래스 속성으로 저장
            self.preprocessor_['cat_cols'] = cat_cols
            self.preprocessor_['cont_cols'] = cont_cols
            self.preprocessor_['encoders'] = encoders
            self.preprocessor_['scaler'] = scaler

            # 훈련/검증 데이터에 전처리 적용
            X_val_df = pd.DataFrame(X_val, columns=feature_names)
            for c, enc in encoders.items():
                X_val_df[c] = X_val_df[c].astype(str).apply(lambda x: x if x in enc.classes_ else 'unknown')
                if 'unknown' not in enc.classes_: enc.classes_ = np.append(enc.classes_, 'unknown')
                X_val_df[c] = enc.transform(X_val_df[c])

            if scaler and cont_cols:
                X_val_df[cont_cols] = scaler.transform(X_val_df[cont_cols])

            # 최종 데이터 준비
            X_cat_train = X_train_df[cat_cols].values if cat_cols else np.empty((len(X_train_df), 0))
            X_cont_train = X_train_df[cont_cols].values if cont_cols else np.empty((len(X_train_df), 0))
            X_cat_val = X_val_df[cat_cols].values if cat_cols else np.empty((len(X_val_df), 0))
            X_cont_val = X_val_df[cont_cols].values if cont_cols else np.empty((len(X_val_df), 0))
            categories = [len(e.classes_) for e in encoders.values()]

            # --- 2. 모델 훈련 ---
            self.model = get_model('FTT', self.model_params, categories=categories, num_continuous=len(cont_cols))
            self.model.to(self.device)

            train_ds = TabularDataset(X_cat_train, X_cont_train, y_train)
            val_ds = TabularDataset(X_cat_val, X_cont_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=self.model_params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.model_params['batch_size'])

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_params['learning_rate'],
                                          weight_decay=self.model_params['weight_decay'])
            criterion = nn.L1Loss()

            best_mae, patience = float('inf'), 0
            train_mae_history, val_mae_history = [], []

            best_model_save_path.parent.mkdir(parents=True, exist_ok=True)

            epoch_pbar = tqdm(range(self.model_params['epochs']), desc=f"Training FTT Epochs")
            for epoch in epoch_pbar:
                self.model.train()
                train_preds_for_epoch, train_true_for_epoch = [], []
                for cat, cont, target_y in train_loader:
                    optimizer.zero_grad()
                    out = self.model(cat.to(self.device), cont.to(self.device))
                    loss = criterion(out, target_y.to(self.device))
                    loss.backward()
                    clip_val = self.model_params.get('gradient_clip_val', None)
                    if clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
                    optimizer.step()
                    train_preds_for_epoch.extend(out.detach().cpu().numpy())
                    train_true_for_epoch.extend(target_y.cpu().numpy())
                train_mae_history.append(mean_absolute_error(train_true_for_epoch, train_preds_for_epoch))

                self.model.eval()
                val_preds_list = []
                with torch.no_grad():
                    for cat, cont, _ in val_loader:
                        val_preds_list.extend(self.model(cat.to(self.device), cont.to(self.device)).cpu().numpy())
                curr_val_mae = mean_absolute_error(y_val, val_preds_list)
                val_mae_history.append(curr_val_mae)

                if curr_val_mae < best_mae:
                    best_mae = curr_val_mae
                    patience = 0
                    torch.save(self.model.state_dict(), best_model_path)
                else:
                    patience += 1
                if patience >= self.model_params['early_stopping_patience']:
                    break

            self.history_ = {'train_mae': train_mae_history, 'val_mae': val_mae_history}

            # 훈련이 끝나면 최고 성능의 모델 가중치를 로드
            self.model.load_state_dict(torch.load(best_model_path))
            return self

        def predict(self, X, feature_names):
            """
            새로운 데이터를 받아 저장된 전처리 규칙으로 변환 후 예측합니다.

            :param X: 예측할 원본 특성 데이터 (numpy array)
            :param feature_names: 특성 이름 리스트
            :return: 예측 결과 (numpy array)
            """
            if not self.preprocessor_ or not self.model:
                raise RuntimeError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

            if self.selector_:
                X = self.selector_.transform(X)
                # 특성 이름도 동일한 규칙으로 업데이트합니다.
                feature_names = [name for name, keep in zip(feature_names, self.selector_.get_support()) if keep]

            # --- 1. 저장된 전처리기로 데이터 변환 ---
            X_df = pd.DataFrame(X, columns=feature_names)
            cat_cols = self.preprocessor_['cat_cols']
            cont_cols = self.preprocessor_['cont_cols']

            for c, enc in self.preprocessor_['encoders'].items():
                known_classes = set(enc.classes_)
                # 테스트 데이터의 값을 문자열로 변환하여 처리
                test_values_str = X_df[c].astype(str)
                # 훈련 시 보지 못했던 새로운 값은 'unknown'으로 처리
                X_df[c] = test_values_str.apply(lambda x: x if x in known_classes else 'unknown')
                # 'unknown'이 LabelEncoder에 없다면 추가
                if 'unknown' not in known_classes:
                    enc.classes_ = np.append(enc.classes_, 'unknown')
                X_df[c] = enc.transform(X_df[c])

            if self.preprocessor_['scaler'] and cont_cols:
                X_df[cont_cols] = self.preprocessor_['scaler'].transform(X_df[cont_cols])

            X_cat_test = X_df[cat_cols].values if cat_cols else np.empty((len(X_df), 0))
            X_cont_test = X_df[cont_cols].values if cont_cols else np.empty((len(X_df), 0))

            # --- 2. 예측 수행 ---
            self.model.eval()
            test_ds = TabularDataset(X_cat_test, X_cont_test)
            test_loader = DataLoader(test_ds, batch_size=self.model_params['batch_size'], shuffle=False)

            preds_list = []
            with torch.no_grad():
                for cat, cont in test_loader:
                    preds_list.extend(self.model(cat.to(self.device), cont.to(self.device)).cpu().numpy())

            return np.array(preds_list).flatten()

    def augment_dataset_gmm(X, y, n_samples=GMM_SAMPLES, n_components=GMM_COMPONENTS, random_state=GMM_RANDOM_STATE):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame or a NumPy array")

        # 컬럼명을 전부 문자열로 변환
        X.columns = X.columns.astype(str)

        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        elif not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series or a NumPy array")

        # 타겟을 추가하고 학습
        df = X.copy()
        df['Target'] = y.values

        # GMM 학습 및 샘플링
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(df)

        synthetic_data, _ = gmm.sample(n_samples)
        synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

        # 원본 + 합성 합치기
        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

        X_augmented = augmented_df.drop(columns='Target')
        y_augmented = augmented_df['Target']

        return X_augmented, y_augmented

    # ------------------------------------------------------------
    # 📊 시각화 함수
    # ------------------------------------------------------------

    def visualize_fold_results(y_val, preds, target, model_name, fold, history=None, show_plot=False):
        """모든 모델의 성능을 시각화하고 그래프를 저장합니다.
        (ET 및 기타 트리 모델은 STEP 간격으로 학습 곡선을 표시)
        """
        residuals = y_val - preds
        has_history = history and ('train_metric' in history or 'train_mae' in history)

        # 모든 모델에 대해 2x2 또는 1x2 레이아웃을 기본으로 설정
        fig, axes = plt.subplots(2, 2, figsize=(14, 10)) if has_history else plt.subplots(1, 2, figsize=(14, 5))
        axes = np.ravel(axes)

        fig.suptitle(f"Results for {target} - {model_name} (Fold {fold + 1})", fontsize=16)

        # 1. 실제값 vs 예측값 산점도 (axes[0])
        axes[0].scatter(y_val, preds, alpha=0.6)
        min_val, max_val = min(y_val.min(), preds.min()), max(y_val.max(), preds.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        axes[0].set(xlabel=f'Actual {target}', ylabel=f'Predicted {target}', title='Actual vs. Predicted')
        axes[0].grid(True, alpha=0.5);
        axes[0].legend()

        # 2. 잔차 분포 히스토그램 (axes[1])
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(residuals.mean(), color='r', ls='--', lw=2, label=f"Mean: {residuals.mean():.2f}")
        axes[1].set(xlabel='Residuals (Actual - Predicted)', ylabel='Frequency', title='Residuals Distribution')
        axes[1].grid(True, alpha=0.5);
        axes[1].legend()

        # 3. 학습 곡선 시각화 (axes[2], axes[3])
        if has_history:
            metric_name = history.get('metric_name', 'Metric').upper()
            tree_models = ['LGBM', 'XGB', 'CAT']

            # --- ET 모델을 위한 분리형 학습 곡선 ---
            if model_name == 'ET':
                ax2, ax3 = axes[2], axes[3]
                x_axis_values = np.arange(1, len(history['train_metric']) + 1) * STEP

                # 왼쪽 아래: Validation MAE 곡선
                ax2.plot(x_axis_values, history['val_metric'], '-o', label=f'Validation {metric_name}', markersize=4,
                         color='tab:orange')
                ax2.set_title('Validation MAE Curve')
                ax2.set_xlabel('Number of Estimators')
                ax2.set_ylabel(metric_name)
                ax2.grid(True, alpha=0.5)
                ax2.legend()

                # 오른쪽 아래: Train MAE 곡선
                ax3.plot(x_axis_values, history['train_metric'], '-o', label=f'Train {metric_name}', markersize=4,
                         color='tab:blue')
                ax3.set_title('Performance on Train Set (MAE)')
                ax3.set_xlabel('Number of Estimators')
                ax3.set_ylabel(metric_name)
                ax3.grid(True, alpha=0.5)
                ax3.legend()

            # --- 그 외 모델을 위한 통합형 학습 곡선 ---
            else:
                ax2 = axes[2]
                # FTT 모델 (모든 Epoch 표시)
                if 'train_mae' in history:
                    epochs = range(1, len(history['train_mae']) + 1)
                    ax2.plot(epochs, history['train_mae'], '-o', label='Train MAE', markersize=4)
                    ax2.plot(epochs, history['val_mae'], '-o', label='Validation MAE', markersize=4)
                    ax2.set(xlabel='Epoch', ylabel='MAE', title='Train vs. Validation MAE Curve')

                # 다른 트리 모델 (LGBM, XGB, CAT)
                elif model_name in tree_models:
                    # STEP 간격으로 데이터를 다운샘플링합니다.
                    train_metric_stepped = history['train_metric'][::STEP]
                    val_metric_stepped = history['val_metric'][::STEP]

                    # 샘플링된 데이터에 맞는 x축 생성 (실제 이터레이션 번호 반영)
                    x_axis_stepped = (np.arange(len(train_metric_stepped)) + 1) * STEP

                    # 그래프 그리기
                    ax2.plot(x_axis_stepped, train_metric_stepped, '-o', label=f'Train {metric_name}', markersize=4)
                    ax2.plot(x_axis_stepped, val_metric_stepped, '-o', label=f'Validation {metric_name}', markersize=4)
                    ax2.set(xlabel='Iteration', ylabel=metric_name, title=f'Train vs. Validation Curve (Step={STEP})')

                ax2.grid(True, alpha=0.5)
                ax2.legend()
                # 네 번째 서브플롯은 사용하지 않으므로 비워둠
                if len(axes) > 3:
                    axes[3].axis('off')

        # 전체 레이아웃 자동 조정 및 저장
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = FOLD_GRAPH_SAVE_DIR / f"{target}_{model_name}_fold{fold + 1}_graphs.png"
        plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close(fig)
        print(f"✅ Visualization for Fold {fold + 1} saved to: {save_path}")

    def visualize_overall_results(y_true, y_pred, train_histories, val_histories, target, model_name, step_size=1,
                                  show_plot=False):
        """
        OOF 예측 결과와 전체 학습 곡선을 하나의 Figure에 종합하여 시각화합니다.
        - 상단: OOF 실제값 vs 예측값, OOF 잔차 분포
        - 하단: 전체 Fold의 평균 학습 곡선 (ET/LGBM/XGB/CAT 모델은 STEP 적용, FTT는 그대로)
        """
        oof_mae = mean_absolute_error(y_true, y_pred)

        # 2x2 서브플롯 레이아웃 생성
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Overall Summary for {target} - {model_name} | OOF MAE: {oof_mae:.5f}', fontsize=20)

        # --- 상단: OOF 결과 시각화 ---
        # 1. (상단 좌측) 실제값 vs OOF 예측값
        ax_oof_scatter = axes[0, 0]
        residuals = y_true - y_pred
        ax_oof_scatter.scatter(y_true, y_pred, alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax_oof_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax_oof_scatter.set_title('Actual vs. OOF Predicted', fontsize=14)
        ax_oof_scatter.set_xlabel(f'Actual {target}', fontsize=12)
        ax_oof_scatter.set_ylabel(f'OOF Predicted {target}', fontsize=12)
        ax_oof_scatter.grid(True, alpha=0.5)
        ax_oof_scatter.legend()

        # 2. (상단 우측) OOF 잔차 분포
        ax_oof_hist = axes[0, 1]
        ax_oof_hist.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax_oof_hist.axvline(residuals.mean(), color='r', ls='--', lw=2, label=f"Mean: {residuals.mean():.2f}")
        ax_oof_hist.set_title('OOF Residuals Distribution', fontsize=14)
        ax_oof_hist.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
        ax_oof_hist.set_ylabel('Frequency', fontsize=12)
        ax_oof_hist.grid(True, alpha=0.5)
        ax_oof_hist.legend()

        # --- 하단: 전체 학습 곡선 시각화 ---
        try:
            min_len = min(len(h) for h in train_histories)
            train_histories_padded = [h[:min_len] for h in train_histories]
            val_histories_padded = [h[:min_len] for h in val_histories]

            mean_train_mae = np.mean(train_histories_padded, axis=0)
            std_train_mae = np.std(train_histories_padded, axis=0)
            mean_val_mae = np.mean(val_histories_padded, axis=0)
            std_val_mae = np.std(val_histories_padded, axis=0)

            # 모델 종류에 따라 학습 곡선 표시 방법을 분기합니다.
            tree_models_with_step = ['LGBM', 'XGB', 'CAT']  # ET는 아래에서 별도 처리

            # 1. ET 모델: Train/Validation 곡선을 분리하여 표시
            if model_name == 'ET':
                x_axis = np.arange(1, len(mean_val_mae) + 1) * step_size

                # (하단 좌측) Validation 곡선
                ax_val_curve = axes[1, 0]
                ax_val_curve.plot(x_axis, mean_val_mae, 'o-', color='tab:orange', label='Average Validation MAE',
                                  markersize=4)
                ax_val_curve.fill_between(x_axis, mean_val_mae - std_val_mae, mean_val_mae + std_val_mae,
                                          color='tab:orange', alpha=0.15)
                best_idx = np.argmin(mean_val_mae)
                ax_val_curve.scatter(x_axis[best_idx], mean_val_mae[best_idx], color='red', s=100, zorder=5,
                                     label=f'Best Val MAE: {mean_val_mae[best_idx]:.5f}')
                ax_val_curve.set_title(f'Overall Validation Curve (Avg over {len(val_histories)} Folds)', fontsize=14)
                ax_val_curve.set_xlabel('Number of Estimators', fontsize=12)
                ax_val_curve.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
                ax_val_curve.legend()
                ax_val_curve.grid(True, alpha=0.5)

                # (하단 우측) Train 곡선
                ax_train_curve = axes[1, 1]
                ax_train_curve.plot(x_axis, mean_train_mae, 'o-', color='tab:blue', label='Average Train MAE',
                                    markersize=4)
                ax_train_curve.fill_between(x_axis, mean_train_mae - std_train_mae, mean_train_mae + std_train_mae,
                                            color='tab:blue', alpha=0.15)
                ax_train_curve.set_title(f'Overall Train Performance (Avg over {len(train_histories)} Folds)',
                                         fontsize=14)
                ax_train_curve.set_xlabel('Number of Estimators', fontsize=12)
                ax_train_curve.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
                ax_train_curve.legend()
                ax_train_curve.grid(True, alpha=0.5)

            # 2. 그 외 트리 모델 (LGBM, XGB, CAT): STEP 간격으로 점을 찍어 통합 표시
            elif model_name in tree_models_with_step:
                ax_lr_curve = axes[1, 0]

                # STEP 간격으로 데이터를 다운샘플링
                plot_train_mae = mean_train_mae[::step_size]
                plot_val_mae = mean_val_mae[::step_size]
                plot_std_train = std_train_mae[::step_size]
                plot_std_val = std_val_mae[::step_size]

                # 샘플링된 데이터에 맞는 x축 생성 (실제 이터레이션 번호 반영)
                plot_x_axis = np.arange(1, len(plot_train_mae) + 1) * step_size

                # 그래프 그리기
                ax_lr_curve.plot(plot_x_axis, plot_train_mae, 'o-', color='tab:blue', label='Average Train MAE',
                                 markersize=4)
                ax_lr_curve.plot(plot_x_axis, plot_val_mae, 'o-', color='tab:orange', label='Average Validation MAE',
                                 markersize=4)
                ax_lr_curve.fill_between(plot_x_axis, plot_train_mae - plot_std_train, plot_train_mae + plot_std_train,
                                         color='tab:blue', alpha=0.15)
                ax_lr_curve.fill_between(plot_x_axis, plot_val_mae - plot_std_val, plot_val_mae + plot_std_val,
                                         color='tab:orange', alpha=0.15)

                # Best MAE 지점 표시 (샘플링된 데이터 기준)
                best_train_idx = np.argmin(plot_train_mae);
                best_val_idx = np.argmin(plot_val_mae)
                ax_lr_curve.scatter(plot_x_axis[best_train_idx], plot_train_mae[best_train_idx], color='blue', s=100,
                                    zorder=5, edgecolor='black',
                                    label=f'Best Train MAE: {plot_train_mae[best_train_idx]:.5f}')
                ax_lr_curve.scatter(plot_x_axis[best_val_idx], plot_val_mae[best_val_idx], color='darkorange', s=100,
                                    zorder=5, edgecolor='black',
                                    label=f'Best Val MAE: {plot_val_mae[best_val_idx]:.5f}')

                ax_lr_curve.set_title(f'Overall Learning Curve (Avg over {len(train_histories)} Folds)', fontsize=14)
                ax_lr_curve.set_xlabel('Iteration', fontsize=12)
                ax_lr_curve.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
                ax_lr_curve.legend(fontsize='small')
                ax_lr_curve.grid(True, which='both', linestyle='--', linewidth=0.5)
                axes[1, 1].axis('off')

            # 3. FTT 모델: 모든 Epoch을 표시
            else:
                ax_lr_curve = axes[1, 0]
                x_axis = np.arange(1, min_len + 1)

                # (FTT 모델의 기존 그래프 로직과 동일)
                ax_lr_curve.plot(x_axis, mean_train_mae, 'o-', color='tab:blue', label='Average Train MAE',
                                 markersize=4)
                ax_lr_curve.plot(x_axis, mean_val_mae, 'o-', color='tab:orange', label='Average Validation MAE',
                                 markersize=4)
                ax_lr_curve.fill_between(x_axis, mean_train_mae - std_train_mae, mean_train_mae + std_train_mae,
                                         color='tab:blue', alpha=0.15)
                ax_lr_curve.fill_between(x_axis, mean_val_mae - std_val_mae, mean_val_mae + std_val_mae,
                                         color='tab:orange',
                                         alpha=0.15)

                best_train_idx = np.argmin(mean_train_mae);
                best_val_idx = np.argmin(mean_val_mae)
                ax_lr_curve.scatter(x_axis[best_train_idx], mean_train_mae[best_train_idx], color='blue', s=100,
                                    zorder=5,
                                    edgecolor='black', label=f'Best Train MAE: {mean_train_mae[best_train_idx]:.5f}')
                ax_lr_curve.scatter(x_axis[best_val_idx], mean_val_mae[best_val_idx], color='darkorange', s=100,
                                    zorder=5,
                                    edgecolor='black', label=f'Best Val MAE: {mean_val_mae[best_val_idx]:.5f}')

                ax_lr_curve.set_title(f'Overall Learning Curve (Avg over {len(train_histories)} Folds)', fontsize=14)
                ax_lr_curve.set_xlabel('Epoch', fontsize=12)
                ax_lr_curve.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
                ax_lr_curve.legend(fontsize='small')
                ax_lr_curve.grid(True, which='both', linestyle='--', linewidth=0.5)
                axes[1, 1].axis('off')

        except ValueError:
            axes[1, 0].text(0.5, 0.5, 'No history data available.', ha='center', va='center')
            axes[1, 1].axis('off')
            print("⚠️ 학습 기록이 없어 전체 학습 곡선 시각화를 건너뜁니다.")

        # 전체 레이아웃 조정 및 저장
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = GRAPH_SAVE_DIR / f"{target}_{model_name}_overall_summary_graphs.png"
        plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close(fig)
        print(f"✅ Overall summary visualization for '{model_name}' saved to: {save_path}")

    # ------------------------------------------------------------
    # 🚀 훈련 모드
    # ------------------------------------------------------------
    MODEL_PIPELINE_MODELS = ['ET', 'LGBM', 'XGB', 'CAT']

    if RUN_MODEL_TRAINING:
        # --- 훈련 및 검증 모드 ---
        print("Starting in [Training & Validation Mode]...")

        overall_oof_scores = defaultdict(dict)

        # 1. 메인 루프: 타겟별 훈련
        for target in TARGETS:
            models_to_train = MODEL_CONFIG.get(target, [])
            if not isinstance(models_to_train, list):
                models_to_train = [models_to_train]
            if not models_to_train:
                continue

            X_path, y_path = INPUT_DIR / f"X_train_{target}.npy", INPUT_DIR / f"y_train_{target}.npy"
            if not (X_path.exists() and y_path.exists()):
                print(f"⚠️ Files not found for target '{target}'. Skipping.");
                continue
            X, y = np.load(X_path), np.load(y_path)

            # 원본 feature_names를 루프 밖에서 한 번만 로드하여 불필요한 중복을 제거합니다.
            original_feature_names = [f'f_{i}' for i in range(X.shape[1])]
            feature_map_path = INPUT_DIR / f"all_feature_names_{target}.pkl"
            if feature_map_path.exists():
                try:
                    original_feature_names = joblib.load(feature_map_path)
                    print(f"INFO: Successfully loaded original feature names for target '{target}'.")
                except Exception as e:
                    print(f"WARNING: Could not load feature names for '{target}'. Using default names. Error: {e}")
            else:
                print(f"INFO: Feature name map not found for '{target}'. Using default 'f_num' names.")

            # 2. 중첩 루프: 모델별 훈련
            for model_name in models_to_train:
                try:
                    print(f"\n{'=' * 60}\n🎯 Training Target: '{target}' using Model: '{model_name}'\n{'=' * 60}")

                    oof_preds, fold_scores, fold_importances = np.zeros(len(y)), [], []
                    all_train_histories, all_val_histories = [], []

                    splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y))

                    # 3. Fold 루프: 교차 검증
                    for fold, (train_idx, val_idx) in enumerate(
                            tqdm(splits, total=len(splits), desc=f"CV for {target}-{model_name}")):
                        history, y_val = None, y[val_idx]
                        X_train, y_train, X_val = X[train_idx], y[train_idx], X[val_idx]

                        # 1. 안전한 데이터 범위를 정의합니다 (32비트 실수 기준).
                        finfo = np.finfo(np.float32)

                        # 2. NaN과 무한대(inf) 값을 안전한 숫자로 변환합니다.
                        X_train = np.nan_to_num(X_train, nan=0.0, posinf=finfo.max, neginf=finfo.min)
                        X_val = np.nan_to_num(X_val, nan=0.0, posinf=finfo.max, neginf=finfo.min)

                        # 3. "너무 큰 값"을 포함한 모든 값을 안전한 범위 내로 강제 제한(clip)합니다.
                        X_train = np.clip(X_train, finfo.min, finfo.max)
                        X_val = np.clip(X_val, finfo.min, finfo.max)

                        # 현재 Fold에서 사용할 feature_names를 원본에서 복사하여 오염을 방지합니다.
                        current_feature_names = original_feature_names.copy()

                        # GMM 증강
                        if DO_GMM_AUGMENT:
                            # DataFrame 생성 시, 오염되지 않은 current_feature_names를 사용합니다.
                            X_train_df = pd.DataFrame(X_train, columns=current_feature_names)
                            X_train_df, y_train_series = augment_dataset_gmm(
                                X_train_df, pd.Series(y_train),
                                n_samples=GMM_SAMPLES, n_components=GMM_COMPONENTS, random_state=GMM_RANDOM_STATE
                            )
                            # 증강 후, 현재 Fold의 변수들만 업데이트합니다.
                            current_feature_names = X_train_df.columns.tolist()
                            X_train = X_train_df.values
                            y_train = y_train_series.values

                        # --- 모델 종류에 따른 분기 ---
                        if model_name == 'FTT':
                            ftt_pipeline = FTTWrapper(model_params=HPARAMS['FTT'], device=DEVICE)
                            best_model_path = FTT_DETAIL_SAVE_DIR / f'best_model_{target}_{model_name}_fold{fold}.pt'
                            ftt_pipeline.fit(
                                X_train, y_train, X_val, y_val,
                                feature_names=current_feature_names,
                                cat_threshold=CATEGORICAL_THRESHOLD,
                                best_model_save_path=best_model_path,
                                do_variance_threshold=DO_VARIANCE_THRESHOLD,
                                variance_threshold_val=VARIANCE_THRESHOLD,
                                do_standard_scaler=DO_StandardScaler
                            )
                            pipeline_path = MODEL_SAVE_DIR / f"ftt_pipeline_{target}_{model_name}_fold{fold}.pkl"
                            joblib.dump(ftt_pipeline, pipeline_path)
                            # 예측 시에는 항상 오염되지 않은 원본 스키마를 사용합니다.
                            preds = ftt_pipeline.predict(X_val, feature_names=original_feature_names)
                            history = ftt_pipeline.history_

                        elif model_name in MODEL_PIPELINE_MODELS:
                            # 1. 파이프라인 단계 동적 구성
                            pipeline_steps = []
                            if DO_VARIANCE_THRESHOLD:
                                pipeline_steps.append(('selector', VarianceThreshold(threshold=VARIANCE_THRESHOLD)))
                            if DO_StandardScaler:
                                pipeline_steps.append(('scaler', StandardScaler()))
                            pipeline_steps.append(('model', get_model(model_name, HPARAMS[model_name])))

                            pipeline = Pipeline(pipeline_steps)

                            # --- 훈련 로직을 2단계로 분리 ---

                            # 2. 1단계: 파이프라인 전체 훈련 (제출 및 공식 예측용)
                            print(f"INFO: Fitting pipeline for {model_name}...")
                            pipeline.fit(X_train, y_train)
                            preds = pipeline.predict(X_val)

                            # 3. 2단계: 학습 곡선 생성을 위한 별도 훈련
                            history = {}  # 기본값으로 빈 history 초기화

                            if model_name == 'ET':
                                print(f"INFO: Generating learning curve for ET by iterative fitting...")

                                # 시각화 전용 임시 파이프라인을 생성하여 warm_start 충돌 방지
                                diag_pipeline_steps = []
                                if DO_VARIANCE_THRESHOLD:
                                    diag_pipeline_steps.append(
                                        ('selector', VarianceThreshold(threshold=VARIANCE_THRESHOLD)))
                                if DO_StandardScaler:
                                    diag_pipeline_steps.append(('scaler', StandardScaler()))

                                et_diag_params = HPARAMS['ET'].copy()
                                et_diag_params['warm_start'] = True
                                et_diag_params.pop('oob_score', None)

                                diag_model = get_model(model_name, et_diag_params)
                                diag_pipeline_steps.append(('model', diag_model))
                                diagnostic_pipeline = Pipeline(diag_pipeline_steps)

                                train_mae_history, val_mae_history = [], []
                                n_estimators_total = HPARAMS['ET']['n_estimators']
                                estimator_range = range(STEP, n_estimators_total + 1, STEP)

                                for n_estimators in tqdm(estimator_range, desc=f"LC for {model_name}", leave=False):
                                    diagnostic_pipeline.set_params(model__n_estimators=n_estimators)
                                    diagnostic_pipeline.fit(X_train, y_train)
                                    train_pred = diagnostic_pipeline.predict(X_train)
                                    val_pred = diagnostic_pipeline.predict(X_val)
                                    train_mae_history.append(mean_absolute_error(y_train, train_pred))
                                    val_mae_history.append(mean_absolute_error(y_val, val_pred))
                                history = {'train_metric': train_mae_history, 'val_metric': val_mae_history,
                                           'metric_name': 'MAE'}

                            elif model_name in ['LGBM', 'XGB', 'CAT']:
                                # 이 모델들은 eval_set을 사용하여 학습 곡선을 효율적으로 생성합니다.
                                # 시각화는 제출용 파이프라인에 영향을 주지 않는 별도 모델로 수행합니다.
                                try:
                                    print(f"INFO: Generating learning curve for {model_name} (visualization only)...")

                                    # 1. 데이터 전처리: 훈련된 파이프라인에서 모델을 제외한 전처리기만 추출
                                    preprocessor_steps = [(name, step) for name, step in pipeline.named_steps.items() if
                                                          name != 'model']

                                    if preprocessor_steps:
                                        # 전처리기만으로 임시 파이프라인을 구성하여 데이터 변환
                                        preprocessor = Pipeline(preprocessor_steps)
                                        X_train_transformed = preprocessor.transform(X_train)
                                        X_val_transformed = preprocessor.transform(X_val)
                                    else:
                                        # 전처리 단계가 없는 경우 원본 데이터 사용
                                        X_train_transformed = X_train
                                        X_val_transformed = X_val

                                    # 2. 시각화용 임시 모델 생성 및 훈련
                                    diagnostic_params = HPARAMS[model_name].copy()
                                    # early_stopping_rounds는 fit 메서드에 전달해야 하므로 생성자 파라미터에서 제거
                                    early_stopping_rounds = diagnostic_params.pop('early_stopping_rounds', 100)
                                    diagnostic_model = get_model(model_name, diagnostic_params)

                                    # eval_set과 함께 fit을 호출하여 학습 곡선 데이터(history) 획득
                                    fit_params = {
                                        'eval_set': [(X_train_transformed, y_train), (X_val_transformed, y_val)]
                                    }
                                    if 'early_stopping_rounds' in HPARAMS[model_name]:
                                        fit_params['early_stopping_rounds'] = early_stopping_rounds

                                    if model_name == 'LGBM':
                                        fit_params['eval_names'] = ['train', 'valid']

                                    elif model_name in ['XGB', 'CAT']:
                                        fit_params['verbose'] = 0

                                    diagnostic_model.fit(X_train_transformed, y_train, **fit_params)

                                    # 3. 학습 기록 추출
                                    history = get_tree_model_history(diagnostic_model, model_name)

                                    # 이 블록이 끝나면 시각화용 모델(diagnostic_model)은 자동으로 폐기됩니다.

                                except Exception as e:
                                    print(f"WARNING: Could not generate learning curve for {model_name}. Error: {e}")
                                    history = {}  # 실패 시 history를 빈 딕셔너리로 초기화

                            # 4. 파이프라인 전체를 저장합니다.
                            pipeline_path = MODEL_SAVE_DIR / f"pipeline_{target}_{model_name}_fold{fold}.pkl"
                            joblib.dump(pipeline, pipeline_path)

                            # 5. 특성 중요도 추출
                            try:
                                model_step = pipeline.named_steps['model']
                                if hasattr(model_step, 'feature_importances_'):
                                    final_feature_names = current_feature_names
                                    if 'selector' in pipeline.named_steps:
                                        selector_mask = pipeline.named_steps['selector'].get_support()
                                        final_feature_names = [name for name, keep in
                                                               zip(current_feature_names, selector_mask)
                                                               if keep]
                                    importances = model_step.feature_importances_
                                    if len(final_feature_names) == len(importances):
                                        imp_df = pd.DataFrame({'feature': final_feature_names, 'importance': importances})
                                        fold_importances.append(imp_df)
                                    else:
                                        print(f"WARNING: FI length mismatch after selection.")
                            except Exception as e:
                                print(f"Could not extract feature importances: {e}")

                        # 현재 폴드의 학습 기록을 리스트에 추가
                        if history:
                            if 'train_mae' in history:  # FTT case
                                all_train_histories.append(history['train_mae'])
                                all_val_histories.append(history['val_mae'])
                            elif 'train_metric' in history:  # Tree models case
                                all_train_histories.append(history['train_metric'])
                                all_val_histories.append(history['val_metric'])

                        # 폴드별 성능 계산 및 시각화
                        oof_preds[val_idx], score = preds, mean_absolute_error(y_val, preds)
                        fold_scores.append(score)
                        visualize_fold_results(y_val, preds, target, model_name, fold, history=history, show_plot=False)
                    # Overall 학습 곡선 시각화 함수 호출
                    if all_train_histories and all_val_histories:
                        tree_models = ['ET', 'LGBM', 'XGB', 'CAT']
                        step_size = STEP if model_name in tree_models else 1
                        visualize_overall_results(
                            y_true=y,
                            y_pred=oof_preds,
                            train_histories=all_train_histories,
                            val_histories=all_val_histories,
                            target=target,
                            model_name=model_name,
                            step_size=step_size,
                            show_plot=False  # True로 바꾸면 그래프가 화면에 표시됩니다.
                        )
                    # --- 모델별 결과 저장 (모든 폴드 훈련 완료 후) ---
                    if fold_importances:
                        agg_imp_df = pd.concat(fold_importances).groupby('feature')['importance'].mean().sort_values(
                            ascending=False).reset_index()
                        imp_path = IMPORTANCE_SAVE_DIR / f"importance_{target}_{model_name}_aggregated.csv"
                        agg_imp_df.to_csv(imp_path, index=False)
                        print(f"✅ Aggregated feature importance saved to: {imp_path}")

                    # 모델의 최종 Out-of-Fold MAE 계산 및 저장
                    oof_mae = mean_absolute_error(y, oof_preds)
                    overall_oof_scores[target][model_name] = oof_mae
                    print(
                        f"\n--- Finished for '{target}' | Model: '{model_name}' | Avg Fold MAE: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f} | OOF MAE: {oof_mae:.5f} ---\n")

                    # OOF 예측 결과 저장 (npy 및 상세 csv)
                    np.save(OOF_SAVE_DIR / f"oof_preds_{target}_{model_name}.npy", oof_preds)
                    oof_details_df = pd.DataFrame(
                        {'sample_index': np.arange(len(y)), 'y_true': y, f'y_pred_{model_name}': oof_preds,
                         'error': np.abs(y - oof_preds)}).sort_values(by='error', ascending=False)
                    oof_details_path = OOF_SAVE_DIR / f"oof_details_{target}_{model_name}.csv"
                    oof_details_df.to_csv(oof_details_path, index=False)
                    print(f"✅ Detailed OOF results saved to: {oof_details_path}")

                except Exception as e:
                    # 🔴🔴🔴 EXCEPT 블록: 에러 발생 시 실행되는 부분 🔴🔴🔴
                    # 어떤 모델에서 에러가 발생했는지 명확하게 로깅
                    print(f"\n{'!' * 60}")
                    print(f"🔴 CRITICAL ERROR during training of '{model_name}' for target '{target}'.")
                    print(f"🔴 Error Type: {type(e).__name__}")
                    print(f"🔴 Error Message: {e}")
                    import traceback
                    traceback.print_exc()  # 상세한 에러 위치 추적
                    print(f"{'!' * 60}\n")

                    # 실패를 기록하고 다음 모델로 넘어감
                    overall_oof_scores[target][model_name] = np.nan  # 실패한 모델은 NaN으로 점수 기록
                    continue  # 현재 모델 훈련을 중단하고 for 루프의 다음 모델로 넘어감

                finally:
                    # ✅✅✅ FINALLY 블록: 성공하든 실패하든 항상 실행 ✅✅✅
                    # 메모리 관리를 위해 가비지 컬렉션을 수행합니다.
                    # 이를 통해 실패한 모델이 점유하던 메모리도 정리할 수 있습니다.
                    print(f"--- Cleaning up resources for {model_name} on target {target} ---")

                    # 메모리 관리를 위해 가비지 컬렉션 수행
                    gc.collect()

            # ------------------------------------------------------------
            # 📊 최종 결과 요약 및 파일 저장
            # ------------------------------------------------------------
            print("\n\nAll Training & Analysis Finished!")

            # 1. 파일에 저장할 요약 텍스트를 리스트로 생성
            summary_lines = []
            summary_lines.append("================== Summary of OOF MAE ==================")
            for target, model_scores in overall_oof_scores.items():
                summary_lines.append(f"\n--- Target: {target} ---")
                if not model_scores:
                    summary_lines.append("  No models were trained for this target.")
                else:
                    for model_name, score in model_scores.items():
                        summary_lines.append(f"  Model '{model_name}': {score:.5f}")
            summary_lines.append("========================================================")

            # 2. 리스트를 하나의 문자열로 합치기
            summary_text = "\n".join(summary_lines)

            # 3. 화면에 결과 출력 (기존 기능 유지)
            print(summary_text)

            # 4. 요약 텍스트를 파일에 저장
            try:
                summary_file_path = OOF_SAVE_DIR / "oof_summary.txt"
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print(f"\n✅ OOF MAE summary saved to: {summary_file_path}")
            except Exception as e:
                print(f"\n⚠️ Could not save OOF MAE summary. Error: {e}")

    # ------------------------------------------------------------
    # 🚀 제출 모드
    # ------------------------------------------------------------
    MODEL_PIPELINE_MODELS = ['ET', 'LGBM', 'XGB', 'CAT']

    if not RUN_MODEL_TRAINING:
        print("RUN_MODEL_TRAINING=False. 외부 데이터셋에서 모델 파일을 복사합니다...")

        # MODEL_SAVE_DIR로 필요한 모든 .pkl 파일을 복사
        # (FTT 모델의 경우, FTT_DETAIL_SAVE_DIR로 .pt 파일 복사)
        shutil.copytree(MODEL_INPUT_DIR, MODEL_SAVE_DIR, dirs_exist_ok=True)
        print("✅ 모델 파일 복사 완료.")

    if SUBMISSION_MODE:
        print("\n🚀 Starting in [Submission Mode]...")

        if __name__ == "__main__":
            print("\nmain_test() 함수를 실행합니다...")
            main_test()

        # 샘플 제출 파일(template) 로드해서 컬럼명만 가져오기
        sample_df = pd.read_csv(SAMPLE_SUBMISSION / 'sample_submission.csv')
        cols = sample_df.columns.tolist()  # ['id','Tg','FFV','Tc','Density','Rg']

        # 빈 DataFrame 생성 (행 수는 test_df 기준)
        submission_df = pd.DataFrame(columns=cols, index=range(len(test_df)))

        # id 칼럼만 test_df의 id 로 채우기
        submission_df['id'] = test_df['id'].values

        # 모든 타겟에 대해 예측 수행
        for target in tqdm(TARGETS, desc="Predicting Targets"):
            models_to_predict = MODEL_CONFIG.get(target, [])
            if not isinstance(models_to_predict, list): models_to_predict = [models_to_predict]
            if not models_to_predict: continue

            # Test 데이터 로드 (main_test에서 생성된 raw feature)
            X_test_path = TEST_INPUT_DIR / f"X_test_{target}.npy"
            if not X_test_path.exists():
                print(f"⚠️ Test data file not found for target '{target}'. Skipping.");
                continue
            X_test_raw = np.load(X_test_path)

            # 특성 이름 로딩 (FTTWrapper의 predict 메서드에 필요)
            feature_names = []
            feature_map_path = INPUT_DIR / f"all_feature_names_{target}.pkl"
            if feature_map_path.exists():
                try:
                    feature_names = joblib.load(feature_map_path)
                except Exception:
                    pass

            target_predictions = []

            # 해당 타겟에 대해 설정된 모든 모델로 예측
            for model_name in models_to_predict:
                print(f"  > Predicting with {model_name} for {target}...")

                fold_predictions = []
                num_folds_to_load = N_SPLITS if N_SPLITS >= 2 else 1

                for fold in range(num_folds_to_load):

                    # 모델 종류에 따라 데이터 처리 방식을 분기
                    if model_name == 'FTT':
                        # 1. 훈련된 FTTWrapper 객체 전체를 불러옵니다.
                        pipeline_path = MODEL_SAVE_DIR / f"ftt_pipeline_{target}_{model_name}_fold{fold}.pkl"
                        ftt_pipeline = joblib.load(pipeline_path)

                        # 2. Wrapper의 predict 메서드를 사용하여 예측합니다.
                        #    내부적으로 전처리(인코딩, 스케일링)가 자동으로 수행됩니다.
                        preds = ftt_pipeline.predict(X_test_raw, feature_names=feature_names)
                        fold_predictions.append(preds)

                    elif model_name in MODEL_PIPELINE_MODELS:
                        # 1. 훈련된 파이프라인 객체 전체를 불러옵니다.
                        pipeline_path = MODEL_SAVE_DIR / f"pipeline_{target}_{model_name}_fold{fold}.pkl"
                        pipeline = joblib.load(pipeline_path)

                        # 2. 파이프라인의 predict 메서드를 사용하여 예측합니다.
                        #    내부적으로 VarianceThreshold, StandardScaler가 자동으로 적용됩니다.
                        preds = pipeline.predict(X_test_raw)
                        fold_predictions.append(preds)

                # 모든 Fold의 예측값을 평균
                model_avg_preds = np.mean(fold_predictions, axis=0)
                target_predictions.append(model_avg_preds)

            # 한 타겟에 여러 모델을 사용한 경우, 모델들의 예측값을 다시 평균 (앙상블)
            final_preds_for_target = np.mean(target_predictions, axis=0)
            submission_df[target] = final_preds_for_target

        # 최종 submission 파일 저장
        submission_path = SUBMISSION_SAVE_DIR / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"\n✅ Submission file created successfully at: {submission_path}")

    # =================================================================

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExperiment completed in {elapsed_time:.2f} seconds.")

    # ✅ 결과 요약을 위해 dict 형태로 반환
    result = params.copy()  # 입력 파라미터를 결과에 포함
    result['elapsed_time_sec'] = elapsed_time

    return result


# 모델 설정 (실험 관련 파라미터)
experiment_params = [
    # Row 1
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 2
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": True, "AUG_NUM": 1, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.1, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 3
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 100, "GMM_COMPONENTS": 3, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.01, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 4
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": True, "AUG_NUM": 3, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 500, "GMM_COMPONENTS": 5, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 5
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": True, "AUG_NUM": 5, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.001, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 6
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 7
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": True, "AUG_NUM": 1, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 1000, "GMM_COMPONENTS": 7, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 8
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 100, "GMM_COMPONENTS": 3, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.1, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 9
    {"DO_RDKit": False, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.01, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 10
    {"DO_RDKit": False, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": True, "AUG_NUM": 3, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 11
    {"DO_RDKit": False, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 500, "GMM_COMPONENTS": 5, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 12
    {"DO_RDKit": False, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": True, "AUG_NUM": 5, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 1000, "GMM_COMPONENTS": 7, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.001, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 13
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.1, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 14
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": True, "AUG_NUM": 1, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 15
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": True, "AUG_NUM": 3, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 100, "GMM_COMPONENTS": 3, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 16
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": False, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 500, "GMM_COMPONENTS": 5, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.01, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 17
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 18
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": True, "AUG_NUM": 5, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.001, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 19
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": True, "AUG_NUM": 1, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 1000, "GMM_COMPONENTS": 7, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.1, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 20
    {"DO_RDKit": True, "DO_POLYBERT": False, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 100, "GMM_COMPONENTS": 3, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 21
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.01, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 22
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": True, "AUG_NUM": 3, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 23
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 500, "GMM_COMPONENTS": 5, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.001, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 24
    {"DO_RDKit": False, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": 0, "DO_AUGMENT_SMILES": True, "AUG_NUM": 5, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 1000, "GMM_COMPONENTS": 7, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 25
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 26
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": True, "AUG_NUM": 1, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.1, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 27
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 100, "GMM_COMPONENTS": 3, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.01, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 28
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": True, "AUG_NUM": 3, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 500, "GMM_COMPONENTS": 5, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 10},
    # Row 29
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": True, "AUG_NUM": 5, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 1000, "GMM_COMPONENTS": 7, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": False, "N_SPLITS": 5},
    # Row 30
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": True, "GMM_SAMPLES": 100, "GMM_COMPONENTS": 3, "GMM_RANDOM_STATE": 42, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.001, "DO_StandardScaler": False, "N_SPLITS": 10},
    # Row 31
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "required", "DO_AUGMENT_SMILES": True, "AUG_NUM": 1, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": False, "VARIANCE_THRESHOLD": 0, "DO_StandardScaler": True, "N_SPLITS": 5},
    # Row 32
    {"DO_RDKit": True, "DO_POLYBERT": True, "DO_FINGERPRINT": True, "RDKit_FILTER_MODE": "useless", "DO_AUGMENT_SMILES": False, "AUG_NUM": 0, "DO_GMM_AUGMENT": False, "GMM_SAMPLES": 0, "GMM_COMPONENTS": 0, "GMM_RANDOM_STATE": 0, "DO_VARIANCE_THRESHOLD": True, "VARIANCE_THRESHOLD": 0.1, "DO_StandardScaler": True, "N_SPLITS": 10}
]

# ✅ 모든 실험 결과를 저장할 리스트
all_results = []
MAIN_OUTPUT_DIR = Path("Output")
MAIN_OUTPUT_DIR.mkdir(exist_ok=True)

# ✅ 실험을 반복하기 위한 for loop
# 1. tqdm 객체를 먼저 생성하고, file=sys.stdout 인자를 전달
progress_bar = tqdm(experiment_params, desc="Running Experiments", file=sys.stdout, dynamic_ncols=True)

# 2. 전체 루프를 새로 정의한 컨텍스트 매니저로 감싸
with redirect_to_tqdm():
    for idx, params in enumerate(progress_bar):
        exp_num = idx + 1
        OUTPUT_DIR = MAIN_OUTPUT_DIR / f"Output{exp_num}"
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

        log_filename = OUTPUT_DIR / f"experiment_{exp_num}_log.txt"
        logger = None  # logger 초기화

        try:
            # TeeLogger는 이제 redirect_to_tqdm이 만들어준 환경에서 안전하게 동작
            logger = TeeLogger(str(log_filename), mode='w')

            # ✅ 함수화된 실험 로직 호출
            result = run_single_experiment(params, OUTPUT_DIR)
            result['experiment_id'] = exp_num
            result['status'] = 'Success'
            all_results.append(result)

        except Exception as e:
            # ✅ 에러 발생 시 로그 남기고 계속 진행
            error_message = f"!!!!!! Experiment {exp_num} FAILED !!!!!!\nError: {e}"
            print(error_message)

            # 에러 발생 시 결과 기록
            failed_result = params.copy()
            failed_result['experiment_id'] = exp_num
            failed_result['status'] = 'Failed'
            failed_result['error_message'] = str(e)
            all_results.append(failed_result)
            continue  # 다음 실험으로 넘어감

        finally:
            # ✅ Logger 리소스 정리
            if logger:
                logger.close()

            # 자원 해제
            gc.collect()

# ✅ 모든 실험이 끝난 후, 결과를 하나의 CSV 파일로 저장
print("\n" + "=" * 50)
print("All experiments are complete. Aggregating results...")
if all_results:
    results_df = pd.DataFrame(all_results)

    # 컬럼 순서 정리 (중요한 정보 앞으로)
    core_cols = ['experiment_id', 'status', 'elapsed_time_sec']
    param_cols = [col for col in results_df.columns if col not in core_cols]
    results_df = results_df[core_cols + param_cols]

    summary_path = MAIN_OUTPUT_DIR / "experiment_summary.csv"
    results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Results summary saved to: {summary_path}")
else:
    print("No results to summarize.")