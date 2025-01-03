# BSPM-base-in-Rechorus
使用`Rechorus`框架中实现`BSPM`推荐算法

本仓库为中山大学-机器学习期末大作业

任务：使用`Rechorus`框架实现一篇论文中的推荐算法

本仓库贡献：成功在`Rechorus`框架中实现了`BSPM`算法

#本地部署：

```bash
git clone https://github.com/lang-longmao/BSPM-base-in-Rechorus.git
```

```bash
cd ReChorus
pip install -r requirements.txt
```

```bash
python src\main.py--model_name BSPM --dataset ML_1MTOPK --T_b=1 --T_idL=1 --K_b=1 --K_idl=1 --T_s=1 --K_s=1 --id_beta=0.003 --fashr="rk4" --train 0ctor_dim=30 --solver_shr rk4 --train 0 --sharpening_off False
```

or

```bash
python src\main.py--model_name BSPM --dataset Grocery_and_Gourmet_Food --T_b=2 --T_idL=3 --K_b=2 --K_idl=2 --T_s=1 --K_s=2 --id_beta=0.3 --fashr="rk4" --train 0ctor_dim=50 --solver_shr rk4 --train 0 --sharpening_off False
```
#运行结果：

## Top-k Recommendation on ML_1MTOPK

| Model                                                                                             | HR@5   | NDCG@5 | HR@10     | NDCG@10    | HR@20     | NDCG@20    | HR@50   | NDCG@50 |
|:------------------------------------------------------------------------------------------------- |:------:|:------:|:---------:|:----------:|:---------:|:----------:|:-------:|:-------:|
| BSPM            | 0.3733 | 0.2489 | 0.5418 | 0.3029 | 0.7425 | 0.3538 | 0.9457 | 0.3947 |


## Top-k Recommendation on Grocery_and_Gourmet_Food

| Model                                                                                             | HR@5   | NDCG@5 | HR@10     | NDCG@10    | HR@20     | NDCG@20    | HR@50   | NDCG@50 |
|:------------------------------------------------------------------------------------------------- |:------:|:------:|:---------:|:----------:|:---------:|:----------:|:-------:|:-------:|
| BSPM            | 0.3614 | 0.2486 | 0.4951 | 0.2920 | 0.6125 | 0.3217 | 0.7949 | 0.3579 |



BSPM原论文: Blurring-Sharpening Process Models for Collaborative Filtering(https://arxiv.org/pdf/2211.09324)


Rechorus: https://github.com/THUwangcy/ReChorus
