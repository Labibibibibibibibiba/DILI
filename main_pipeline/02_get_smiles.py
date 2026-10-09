import pandas as pd
import requests
import time


# =========================
# 1. 读取DILI数据
# =========================

data = pd.read_excel("../data/DILI_ML_dataset_v1.xlsx")


print("数据读取成功")
print(data.head())


# =========================
# 2. 定义查询PubChem函数
# =========================

def get_smiles(drug_name):

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"

    try:

        response = requests.get(url, timeout=10)


        if response.status_code == 200:

            result = response.json()

            smiles = result["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]

            return smiles


        else:

            return None


    except Exception:

        return None



# =========================
# 3. 批量获取SMILES
# =========================


smiles_list = []


for i, drug in enumerate(data["CompoundName"]):

    print(f"{i+1}/{len(data)} 正在查询: {drug}")


    smiles = get_smiles(drug)


    smiles_list.append(smiles)


    # 防止访问过快
    time.sleep(0.2)



# =========================
# 4. 保存结果
# =========================

data["SMILES"] = smiles_list


data.to_excel(
    "../data/DILI_ML_dataset_v2.xlsx",
    index=False
)


print("全部完成！")