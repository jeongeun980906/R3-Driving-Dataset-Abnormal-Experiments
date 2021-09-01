import json
import torch

data_name_list = ['0729_exp_gunmin_FMTC'
                ,'0729_exp_gunmin_highway'
                ,'0729_exp_gunmin_road'
                ,'0729_exp_jeongwoo_FMTC'
                ,'0729_exp_sumin_FMTC'
                ,'0729_exp_sumin_highway'
                ,'0729_exp_sumin_road'
                ,'0729_exp_wooseok_FMTC'
                ,'0729_neg_gunmin_01_1'
                ,'0729_neg_gunmin_02_1'
                ,'0729_neg_gunmin_03_1'
                ,'0729_neg_gunmin_05_1'
                ,'0729_neg_gunmin_06_1'
                ,'0729_neg_gunmin_08_1'
                ,'0729_neg_gunmin_09_1'
                ,'0729_neg_gunmin_10_1'
                ,'0729_neg_gunmin_16_1'
                ,'0729_neg_gunmin_16_2'
                ,'0729_neg_gunmin_28_1'
                ,'0729_neg_gunmin_28_2'
                ,'0729_neg_gunmin_29_1'
                ,'0729_neg_gunmin_30_1'
                ,'0729_neg_gunmin_30_2'
                ,'0729_neg_gunmin_31_1'
                ,'0729_neg_gunmin_31_2'
                ,'0729_neg_gunmin_34_1'
                ,'0729_neg_gunmin_34_2'
                ,'0729_neg_gunmin_35_1'
                ,'0729_neg_gunmin_35_2'
                ,'0729_neg_gunmin_36_1'
                ,'0729_neg_gunmin_36_2'
                ,'0729_neg_gunmin_37_1'
                ,'0729_neg_gunmin_37_2'
                ,'0729_neg_gunmin_38_1'
                ,'0729_neg_gunmin_38_2'
                ,'0729_neg_gunmin_38_3'
                ,'0729_neg_gunmin_38_4'
                ,'0729_neg_gunmin_38_5'
                ,'0729_neg_gunmin_38_6'
                ,'0729_neg_gunmin_50_1'
                ,'0729_neg_gunmin_50_2'
                ,'0729_neg_jeongwoo_01_1'
                ,'0729_neg_jeongwoo_02_1'
                ,'0729_neg_jeongwoo_03_1'
                ,'0729_neg_jeongwoo_05_1'
                ,'0729_neg_jeongwoo_06_1'
                ,'0729_neg_jeongwoo_08_1'
                ,'0729_neg_jeongwoo_09_1'
                ,'0729_neg_jeongwoo_10_1'
                ,'0729_neg_jeongwoo_50_1'
                ,'0729_neg_jeongwoo_50_2'
                ,'0729_neg_sumin_01_1'
                ,'0729_neg_sumin_02_1'
                ,'0729_neg_sumin_03_1'
                ,'0729_neg_sumin_05_1'
                ,'0729_neg_sumin_06_1'
                ,'0729_neg_sumin_08_1'
                ,'0729_neg_sumin_09_1'
                ,'0729_neg_sumin_10_1'
                ,'0729_neg_sumin_38_1'
                ,'0729_neg_sumin_38_2'
                ,'0729_neg_sumin_38_3'
                ,'0729_neg_sumin_38_4'
                ,'0729_neg_sumin_42_1'
                ,'0729_neg_sumin_42_2'
                ,'0729_neg_sumin_42_3'
                ,'0729_neg_sumin_42_4'
                ,'0729_neg_sumin_50_1'
                ,'0729_neg_sumin_50_2'
                ,'0729_neg_wooseok_01_1'
                ,'0729_neg_wooseok_02_1'
                ,'0729_neg_wooseok_03_1'
                ,'0729_neg_wooseok_05_1'
                ,'0729_neg_wooseok_06_1'
                ,'0729_neg_wooseok_08_1'
                ,'0729_neg_wooseok_09_1'
                ,'0729_neg_wooseok_10_1'
                ,'0729_neg_wooseok_28'
                ,'0729_neg_wooseok_28_1'
                ,'0729_neg_wooseok_29_1'
                ,'0729_neg_wooseok_29_2'
                ,'0729_neg_wooseok_30_1'
                ,'0729_neg_wooseok_30_2'
                ,'0729_neg_wooseok_31_1'
                ,'0729_neg_wooseok_31_2'
                ,'0729_neg_wooseok_34_2'
                ,'0729_neg_wooseok_35_1'
                ,'0729_neg_wooseok_35_2'
                ,'0729_neg_wooseok_36_1'
                ,'0729_neg_wooseok_36_2'
                ,'0729_neg_wooseok_37_1'
                ,'0729_neg_wooseok_37_2'
                ,'0729_neg_wooseok_46'
                ,'0729_neg_wooseok_47'
                ,'0729_neg_wooseok_50_1'
                ,'0729_neg_wooseok_50_2'
                ,'0813_exp_jeongwoo_road_1'
                ,'0813_exp_jeongwoo_road_2'
                ,'0815_exp_jeongwoo_highway_1'
                ,'0815_exp_jeongwoo_highway_2'
                ,'0826_exp_jeongwoo_FMTC'
                ,'0823_neg_jeongwoo_01_01'
                ,'0823_neg_jeongwoo_01_02'
                ,'0823_neg_jeongwoo_01_03'
                ,'0823_neg_jeongwoo_01_04'
                ,'0823_neg_jeongwoo_01_05'
                ,'0823_neg_jeongwoo_02_01'
                ,'0823_neg_jeongwoo_02_02'
                ,'0823_neg_jeongwoo_02_03'
                ,'0823_neg_jeongwoo_02_04'
                ,'0823_neg_jeongwoo_02_05'
                ,'0823_neg_jeongwoo_02_06'
                ,'0823_neg_jeongwoo_02_07'
                ,'0823_neg_jeongwoo_02_08'
                ,'0823_neg_jeongwoo_02_09'
                ,'0823_neg_jeongwoo_02_10'
                ,'0826_neg_jeongwoo_03_01'
                ,'0826_neg_jeongwoo_03_02'
                ,'0826_neg_jeongwoo_03_04'
                ,'0826_neg_jeongwoo_03_05'
                ,'0826_neg_jeongwoo_04_01'
                ,'0826_neg_jeongwoo_04_02'
                ,'0826_neg_jeongwoo_04_03'
                ,'0826_neg_jeongwoo_04_04'
                ,'0826_neg_jeongwoo_04_05'
                ,'0826_neg_jeongwoo_04_06'
                ,'0826_neg_jeongwoo_04_07'
                ,'0826_neg_jeongwoo_04_08'
                ,'0826_neg_jeongwoo_04_09'
                ,'0826_neg_jeongwoo_04_10'
                ,'0826_neg_jeongwoo_05_01'
                ,'0826_neg_jeongwoo_05_02'
                ,'0826_neg_jeongwoo_05_03'
                ,'0826_neg_jeongwoo_05_04'
                ,'0826_neg_jeongwoo_05_05'
                ,'0826_neg_jeongwoo_05_06'
                ,'0826_neg_jeongwoo_05_07'
                ,'0826_neg_jeongwoo_05_08'
                ,'0826_neg_jeongwoo_05_09'
                ,'0826_neg_jeongwoo_05_10'
                ,'0826_neg_jeongwoo_06_02'
                ,'0826_neg_jeongwoo_06_03'
                ,'0826_neg_jeongwoo_06_04'
                ,'0826_neg_jeongwoo_06_05'
                ,'0823_neg_jeongwoo_09_01'
                ,'0823_neg_jeongwoo_09_02'
                ,'0823_neg_jeongwoo_09_03'
                ,'0823_neg_jeongwoo_09_04'
                ,'0823_neg_jeongwoo_09_05'
                ,'0823_neg_jeongwoo_09_06'
                ,'0823_neg_jeongwoo_09_07'
                ,'0823_neg_jeongwoo_09_08'
                ,'0823_neg_jeongwoo_09_09'
                ,'0823_neg_jeongwoo_09_10'
                ,'0826_neg_jeongwoo_12_01'
                ,'0826_neg_jeongwoo_12_02'
                ,'0826_neg_jeongwoo_12_03'
                ,'0826_neg_jeongwoo_12_04'
                ,'0826_neg_jeongwoo_12_05'
                ,'0826_neg_jeongwoo_13_01'
                ,'0826_neg_jeongwoo_13_02'
                ,'0826_neg_jeongwoo_13_03'
                ,'0826_neg_jeongwoo_13_04'
                ,'0826_neg_jeongwoo_13_05'
                ,'0826_neg_jeongwoo_14_01'
                ,'0826_neg_jeongwoo_14_02'
                ,'0826_neg_jeongwoo_14_03'
                ,'0826_neg_jeongwoo_14_04'
                ,'0826_neg_jeongwoo_14_05'
                ,'0826_neg_jeongwoo_15_01'
                ,'0826_neg_jeongwoo_15_02'
                ,'0826_neg_jeongwoo_15_03'
                ,'0826_neg_jeongwoo_15_04'
                ,'0826_neg_jeongwoo_15_05'
                ,'0826_neg_jeongwoo_16_01'
                ,'0826_neg_jeongwoo_16_02'
                ,'0826_neg_jeongwoo_16_03'
                ,'0826_neg_jeongwoo_16_04'
                ,'0826_neg_jeongwoo_16_05'
                ,'0826_neg_jeongwoo_16_06'
                ,'0826_neg_jeongwoo_16_07'
                ,'0826_neg_jeongwoo_16_08'
                ,'0826_neg_jeongwoo_16_09'
                ,'0826_neg_jeongwoo_16_10'
                ,'0826_neg_jeongwoo_17_01'
                ,'0826_neg_jeongwoo_17_02'
                ,'0826_neg_jeongwoo_17_03'
                ,'0826_neg_jeongwoo_17_04'
                ,'0826_neg_jeongwoo_17_05'
                ,'0826_neg_jeongwoo_18_01'
                ,'0826_neg_jeongwoo_18_02'
                ,'0826_neg_jeongwoo_18_03'
                ,'0826_neg_jeongwoo_18_04'
                ,'0826_neg_jeongwoo_18_05'
                ,'0826_neg_jeongwoo_18_06'
                ,'0826_neg_jeongwoo_19_01'
                ,'0826_neg_jeongwoo_19_02'
                ,'0826_neg_jeongwoo_19_03'
                ,'0826_neg_jeongwoo_19_04'
                ,'0826_neg_jeongwoo_19_05'
                ,'0826_neg_jeongwoo_20_01'
                ,'0826_neg_jeongwoo_20_02'
                ,'0826_neg_jeongwoo_20_03'
                ,'0826_neg_jeongwoo_20_04'
                ,'0826_neg_jeongwoo_20_05'
                ,'0826_neg_jeongwoo_21_01'
                ,'0826_neg_jeongwoo_21_02'
                ,'0826_neg_jeongwoo_21_03'
                ,'0826_neg_jeongwoo_21_04'
                ,'0826_neg_jeongwoo_21_05'
                ,'0826_neg_jeongwoo_23_01'
                ,'0826_neg_jeongwoo_23_02'
                ,'0826_neg_jeongwoo_23_03'
                ,'0826_neg_jeongwoo_23_04'
                ,'0826_neg_jeongwoo_23_05'
                ,'0826_neg_jeongwoo_24_01'
                ,'0826_neg_jeongwoo_24_02'
                ,'0826_neg_jeongwoo_24_03'
                ,'0826_neg_jeongwoo_24_04'
                ,'0826_neg_jeongwoo_24_05'
                ,'0826_neg_jeongwoo_24_06'
                ,'0826_neg_jeongwoo_24_07'
                ,'0826_neg_jeongwoo_25_01'
                ,'0826_neg_jeongwoo_25_02'
                ,'0826_neg_jeongwoo_25_03'
                ,'0826_neg_jeongwoo_25_04'
                ,'0826_neg_jeongwoo_25_05'
                ,'0826_neg_jeongwoo_26_01'
                ,'0826_neg_jeongwoo_26_02'
                ,'0826_neg_jeongwoo_26_03'
                ,'0826_neg_jeongwoo_26_04'
                ,'0826_neg_jeongwoo_26_05'
                ,'0823_neg_jeongwoo_27_01'
                ,'0823_neg_jeongwoo_27_02'
                ,'0823_neg_jeongwoo_27_03'
                ,'0823_neg_jeongwoo_27_04'
                ,'0823_neg_jeongwoo_27_05'
                ,'0823_neg_jeongwoo_27_06'
                ,'0823_neg_jeongwoo_27_07'
                ,'0823_neg_jeongwoo_27_08'
                ,'0823_neg_jeongwoo_27_09'
                ,'0823_neg_jeongwoo_27_10'
                ,'0823_neg_jeongwoo_28_01'
                ,'0823_neg_jeongwoo_28_02'
                ,'0823_neg_jeongwoo_28_03'
                ,'0823_neg_jeongwoo_28_04'
                ,'0823_neg_jeongwoo_28_05'
                ,'0823_neg_jeongwoo_28_06'
                ,'0823_neg_jeongwoo_28_07'
                ,'0823_neg_jeongwoo_28_08'
                ,'0823_neg_jeongwoo_28_09'
                ,'0823_neg_jeongwoo_28_10'
                ,'0823_neg_jeongwoo_29_01'
                ,'0823_neg_jeongwoo_29_02'
                ,'0823_neg_jeongwoo_29_03'
                ,'0823_neg_jeongwoo_29_04'
                ,'0823_neg_jeongwoo_29_05'
                ,'0823_neg_jeongwoo_29_06'
                ,'0823_neg_jeongwoo_29_07'
                ,'0823_neg_jeongwoo_29_08'
                ,'0823_neg_jeongwoo_29_09'
                ,'0823_neg_jeongwoo_29_10'
                ,'0823_neg_jeongwoo_30_01'
                ,'0823_neg_jeongwoo_30_02'
                ,'0823_neg_jeongwoo_30_03'
                ,'0823_neg_jeongwoo_30_04'
                ,'0823_neg_jeongwoo_30_05'
                ,'0823_neg_jeongwoo_30_06'
                ,'0823_neg_jeongwoo_30_07'
                ,'0823_neg_jeongwoo_30_08'
                ,'0823_neg_jeongwoo_30_09'
                ,'0823_neg_jeongwoo_30_10'
                ,'0823_neg_jeongwoo_32_01'
                ,'0823_neg_jeongwoo_32_02'
                ,'0823_neg_jeongwoo_32_03'
                ,'0823_neg_jeongwoo_32_04'
                ,'0823_neg_jeongwoo_32_05'
                ,'0823_neg_jeongwoo_32_06'
                ,'0823_neg_jeongwoo_32_07'
                ,'0823_neg_jeongwoo_32_08'
                ,'0823_neg_jeongwoo_32_09'
                ,'0823_neg_jeongwoo_32_10'
                ,'0823_neg_jeongwoo_33_01'
                ,'0823_neg_jeongwoo_33_02'
                ,'0823_neg_jeongwoo_33_03'
                ,'0823_neg_jeongwoo_33_04'
                ,'0823_neg_jeongwoo_33_05'
                ,'0823_neg_jeongwoo_33_06'
                ,'0823_neg_jeongwoo_33_07'
                ,'0823_neg_jeongwoo_33_08'
                ,'0823_neg_jeongwoo_33_09'
                ,'0823_neg_jeongwoo_33_10'
                ,'0823_neg_jeongwoo_34_01'
                ,'0823_neg_jeongwoo_34_02'
                ,'0823_neg_jeongwoo_34_03'
                ,'0823_neg_jeongwoo_34_04'
                ,'0823_neg_jeongwoo_34_05'
                ,'0823_neg_jeongwoo_34_06'
                ,'0823_neg_jeongwoo_34_07'
                ,'0823_neg_jeongwoo_34_08'
                ,'0823_neg_jeongwoo_34_09'
                ,'0823_neg_jeongwoo_34_10'
                ,'0823_neg_jeongwoo_35_01'
                ,'0823_neg_jeongwoo_35_02'
                ,'0823_neg_jeongwoo_35_03'
                ,'0823_neg_jeongwoo_35_04'
                ,'0823_neg_jeongwoo_35_05'
                ,'0823_neg_jeongwoo_35_06'
                ,'0823_neg_jeongwoo_35_07'
                ,'0823_neg_jeongwoo_35_08'
                ,'0823_neg_jeongwoo_35_09'
                ,'0823_neg_jeongwoo_35_10'
                ,'0823_neg_jeongwoo_36_01'
                ,'0823_neg_jeongwoo_36_02'
                ,'0823_neg_jeongwoo_36_03'
                ,'0823_neg_jeongwoo_36_04'
                ,'0823_neg_jeongwoo_36_05'
                ,'0823_neg_jeongwoo_36_06'
                ,'0823_neg_jeongwoo_36_07'
                ,'0823_neg_jeongwoo_36_08'
                ,'0823_neg_jeongwoo_36_09'
                ,'0823_neg_jeongwoo_36_10'
                ,'0823_neg_jeongwoo_37_01'
                ,'0823_neg_jeongwoo_37_02'
                ,'0823_neg_jeongwoo_37_03'
                ,'0823_neg_jeongwoo_37_04'
                ,'0823_neg_jeongwoo_37_05'
                ,'0823_neg_jeongwoo_37_06'
                ,'0823_neg_jeongwoo_37_07'
                ,'0823_neg_jeongwoo_37_08'
                ,'0823_neg_jeongwoo_37_09'
                ,'0823_neg_jeongwoo_37_10'
                ,'0823_neg_jeongwoo_38_01'
                ,'0823_neg_jeongwoo_38_02'
                ,'0823_neg_jeongwoo_38_03'
                ,'0823_neg_jeongwoo_38_04'
                ,'0823_neg_jeongwoo_38_05'
                ,'0823_neg_jeongwoo_38_06'
                ,'0823_neg_jeongwoo_38_07'
                ,'0823_neg_jeongwoo_38_08'
                ,'0823_neg_jeongwoo_38_09'
                ,'0823_neg_jeongwoo_38_10'
                ,'0823_neg_jeongwoo_41_01'
                ,'0823_neg_jeongwoo_41_02'
                ,'0823_neg_jeongwoo_41_03'
                ,'0823_neg_jeongwoo_41_04'
                ,'0823_neg_jeongwoo_41_05'
                ,'0823_neg_jeongwoo_41_06'
                ,'0823_neg_jeongwoo_41_07'
                ,'0823_neg_jeongwoo_41_08'
                ,'0823_neg_jeongwoo_41_09'
                ,'0823_neg_jeongwoo_41_10'
                ,'0823_neg_jeongwoo_42_01'
                ,'0823_neg_jeongwoo_42_02'
                ,'0823_neg_jeongwoo_42_03'
                ,'0823_neg_jeongwoo_42_04'
                ,'0823_neg_jeongwoo_42_05'
                ,'0823_neg_jeongwoo_42_06'
                ,'0823_neg_jeongwoo_42_07'
                ,'0823_neg_jeongwoo_42_08'
                ,'0823_neg_jeongwoo_42_09'
                ,'0823_neg_jeongwoo_42_10'
                ,'0823_neg_jeongwoo_43_01'
                ,'0823_neg_jeongwoo_43_02'
                ,'0823_neg_jeongwoo_43_03'
                ,'0823_neg_jeongwoo_43_04'
                ,'0823_neg_jeongwoo_43_05'
                ,'0823_neg_jeongwoo_43_06'
                ,'0823_neg_jeongwoo_43_07'
                ,'0823_neg_jeongwoo_43_08'
                ,'0823_neg_jeongwoo_43_09'
                ,'0823_neg_jeongwoo_43_10'
                ,'0823_neg_jeongwoo_44_01'
                ,'0823_neg_jeongwoo_44_02'
                ,'0823_neg_jeongwoo_44_03'
                ,'0823_neg_jeongwoo_44_04'
                ,'0823_neg_jeongwoo_44_05'
                ,'0823_neg_jeongwoo_44_06'
                ,'0823_neg_jeongwoo_44_07'
                ,'0823_neg_jeongwoo_44_08'
                ,'0823_neg_jeongwoo_44_09'
                ,'0823_neg_jeongwoo_44_10']


seq_list = [(50,2450),
            (6000,9000),
            (5000,8000),
            (0,2200),
            (150,2550),
            (500,3500),
            (6000,9000),
            (50,1450),
            (40,140),
            (0,140),
            (0,160),
            (90,150),
            (90,170),
            (50,150),
            (90,190),
            (30,80),
            (80,180),
            (220,260),
            (200,220),
            (120,140),
            (120,140),
            (130,160),
            (180,200),
            (160,180),
            (145,165),
            (120,140),
            (90,110),
            (140,160),
            (180,200),
            (120,140),
            (180,200),
            (80,100),
            (120,130),
            (120,140),
            (150,170),
            (140,160),
            (150,170),
            (240,270),
            (180,200),
            (100,200),
            (80,200),
            (0,150),
            (0,150),
            (0,100),
            (80,190),
            (110,190),
            (50,150),
            (170,220),
            (70,100),
            (90,160),
            (90,290),
            (80,150),
            (70,160),
            (40,140),
            (40,80),
            (230,270),
            (50,150),
            (50,70),
            (130,150),
            (180,200),
            (120,140),
            (155,175),
            (120,140),
            (190,210),
            (140,160),
            (110,130),
            (130,150),
            (80,250),
            (80,180),
            (80,160),
            (50,150),
            (50,130),
            (120,190),
            (100,150),
            (40,110),
            (300,350),
            (360,400),
            (160,190),
            (125,155),
            (120,140),
            (100,110),
            (80,100),
            (90,110),
            (70,90),
            (110,130),
            (130,150),
            (90,110),
            (90,110),
            (230,250),
            (80,100),
            (100,130),
            (110,140),
            (170,190),
            (160,190),
            (30,130),
            (170,220),
            (1400,4400),
            (14000,17000),
            (1400,4400),
            (5500,8500),
            (500, 10500),
            (130,330),
            (0,200),
            (0,250),
            (0,200),
            (0,250),
            (0,130),
            (0,100),
            (0,130),
            (0,100),
            (0,150),
            (0,100),
            (0,130),
            (0,100),
            (0,130),
            (0,110),
            (40,180),
            (30,160),
            (30,140),
            (30,180),
            (60,100),
            (55,95),
            (50,110),
            (50,100),
            (90,150),
            (50,100),
            (40,120),
            (40,100),
            (90,140),
            (60,100),
            (50,100),
            (40,70),
            (90,140),
            (50,100),
            (70,130),
            (100,150),
            (70,120),
            (60,110),
            (90,140),
            (50,100),
            (50,210),
            (30,260),
            (30,140),
            (0,160),
            (20,170),
            (0,170),
            (0,170),
            (0,170),
            (0,160),
            (0,160),
            (0,150),
            (0,160),
            (0,150),
            (0,170),
            (55,100),
            (90,150),
            (90,140),
            (90,130),
            (95,145),
            (70,110),
            (90,130),
            (90,120),
            (75,125),
            (75,120),
            (100,140),
            (90,110),
            (70,100),
            (60,90),
            (45,80),
            (110,140),
            (130,155),
            (90,120),
            (100,130),
            (100,125),
            (80,110),
            (40,80),
            (100,150),
            (100,140),
            (75,105),
            (100,150),
            (60,95),
            (60,100),
            (80,130),
            (40,90),
            (100,140),
            (70,110),
            (60,95),
            (80,130),
            (70,120),
            (0,250),
            (10,350),
            (10,200),
            (10,240),
            (10,260),
            (10,240),
            (10,300),
            (60,330),
            (30,310),
            (0,300),
            (0,300),
            (95,120),
            (60,90),
            (100,140),
            (110,140),
            (130,160),
            (80,120),
            (100,130),
            (195,220),
            (120,150),
            (140,175),
            (60,100),
            (60,110),
            (60,100),
            (60,100),
            (80,110),
            (90,110),
            (100,130),
            (120,140),
            (105,125),
            (95,110),
            (115,135),
            (150,160),
            (80,105),
            (65,95),
            (100,120),
            (70,90),
            (75,95),
            (75,90),
            (60,70),
            (70,80),
            (55,70),
            (30,50),
            (25,55),
            (45,65),
            (35,55),
            (40,70),
            (40,60),
            (30,50),
            (0,30),
            (40,60),
            (40,60),
            (150,170),
            (20,50),
            (40,70),
            (50,85),
            (90,120),
            (90,110),
            (80,105),
            (95,120),
            (105,130),
            (90,110),
            (90,120),
            (15,45),
            (20,50),
            (30,60),
            (10,50),
            (20,50),
            (35,70),
            (25,55),
            (30,60),
            (30,60),
            (15,55),
            (25,55),
            (25,55),
            (25,50),
            (15,60),
            (0,45),
            (20,35),
            (50,75),
            (35,60),
            (35,65),
            (30,65),
            (30,60),
            (40,75),
            (25,50),
            (55,80),
            (35,65),
            (35,65),
            (35,60),
            (50,85),
            (25,55),
            (30,65),
            (25,50),
            (50,85),
            (20,60),
            (20,50),
            (35,65),
            (20,55),
            (20,50),
            (20,50),
            (30,60),
            (25,60),
            (30,70),
            (30,60),
            (50,90),
            (30,50),
            (60,100),
            (20,50),
            (25,55),
            (50,85),
            (20,50),
            (20,50),
            (70,105),
            (70,115),
            (90,125),
            (70,90),
            (60,90),
            (60,90),
            (100,130),
            (55,85),
            (70,110),
            (55,90),
            (0,40),
            (20,50),
            (20,60),
            (30,70),
            (25,60),
            (25,65),
            (25,60),
            (35,80),
            (0,30),
            (40,65),
            (30,60),
            (10,35),
            (20,50),
            (20,50),
            (20,55),
            (15,40),
            (10,40),
            (15,45),
            (10,40),
            (10,40),
            (15,50),
            (15,70),
            (20,50),
            (20,50),
            (0,35),
            (55,85),
            (20,50),
            (15,40),
            (40,70),
            (40,65),
            (20,40),
            (20,40),
            (20,40),
            (15,40),
            (20,50),
            (20,45),
            (20,40),
            (20,40),
            (20,45),
            (0,35),
            (20,60),
            (40,80),
            (20,40),
            (20,50),
            (20,45),
            (20,60),
            (20,55),
            (20,50),
            (10,40),
            (15,45),
            (65,85),
            (40,80),
            (55,90),
            (40,80),
            (40,80),
            (35,60),
            (55,95),
            (50,65),
            (80,120),
            (55,100),
            (20,45),
            (30,50),
            (20,40),
            (10,40),
            (35,55),
            (40,65),
            (30,50),
            (25,50),
            (30,55),
            (30,50)]

expert_index=[0,1,2,3,4,5,6,7,96,97,98,99,100]
FMTC_index = [0,4,5,7,100]
urban_index = [2,6,96,97]
highway_index = [1,3,98,99]

def check_exp_case(c):
    res = [0]*3
    if c in FMTC_index:
        res[0]=1
    elif c in urban_index:
        res[1]=1
    elif c in highway_index:
        res[2]=1
    return res

negative_index = []
for i in range(0,382):
    if i not in expert_index:
        negative_index.append(i)

straight_road_index = [8,9,10,13,16,17,46,56,74,92,93,94,95]
for i in range(33,44):
    straight_road_index.append(i)
for i in range(49,54):
    straight_road_index.append(i)
for i in range(59,72):
    straight_road_index.append(i)
for i in range(101,120):
    straight_road_index.append(i)
for i in range(140,232):
    straight_road_index.append(i)

cross_road_index = []
for i in negative_index:
    if i not in straight_road_index:
        cross_road_index.append(i)

unstable_index = [13,14,15,39,40,46,47,48,49,50,56,57,58,67,68,74,75,76,94,95]
for i in range(101, 154):
    unstable_index.append(i)

lane_keeping_index = [8,9,10,11,12,94,95]
for i in range(39,46):
    lane_keeping_index.append(i)
for i in range(49,56):
    lane_keeping_index.append(i)
for i in range(67,74):
    lane_keeping_index.append(i)
for i in range(101,120):
    lane_keeping_index.append(i)
for i in range(144,154):
    lane_keeping_index.append(i)

lane_changing_index = []
for i in range(33,39):
    lane_changing_index.append(i)
for i in range(59,63):
    lane_changing_index.append(i)
for i in range(154,164):
    lane_changing_index.append(i)
for i in range(169,189):
    lane_changing_index.append(i)
for i in range(200,205):
    lane_changing_index.append(i)
for i in range(210,215):
    lane_changing_index.append(i)

overtaking_index = []
for i in range(33,39):
    overtaking_index.append(i)
for i in range(59,63):
    overtaking_index.append(i)
for i in range(159,164):
    overtaking_index.append(i)
for i in range(169,174):
    overtaking_index.append(i)
for i in range(184,189):
    overtaking_index.append(i)
for i in range(200,205):
    overtaking_index.append(i)

collision_index = []
for i in range(16,39):
    collision_index.append(i)
for i in range(59,67):
    collision_index.append(i)
for i in range(77,94):
    collision_index.append(i)
for i in range(154,382):
    collision_index.append(i)

def check_neg_case(c):
    res = [0]*7
    if c in straight_road_index:
        res[0]=1
    if c in cross_road_index:
        res[1]=1
    if c in unstable_index:
        res[2]=1
    if c in lane_keeping_index:
        res[3]=1
    if c in lane_changing_index:
        res[4]=1
    if c in overtaking_index:
        res[5]=1
    if c in collision_index:
        res[6]=1
    return res


MAX_N_OBJECTS = 5
def load_expert_dataset(exp_path,exp_case,frame):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + N_OBJECTS * 6)
    """
    rt = []
    act = []
    case = []
    expert_index2=[]
    for e in exp_case:
        if e==1:
            expert_index2.extend(FMTC_index)
        if e==2:
            expert_index2.extend(urban_index)
        if e==3:
            expert_index2.extend(highway_index)
    for data_index in expert_index2:
        data_name = data_name_list[data_index]
        data_path = exp_path + data_name + "/"
        state_path = data_path + "state/"
        st = seq_list[data_index][0]
        en = seq_list[data_index][1]
        for seq in range(st + 1, en + 1 - frame):
            data = []
            data_act = []
            for it in range(frame):
                state_file = state_path + str(seq+it).zfill(6) + ".json"
                with open(state_file, "r") as st_json:
                    state = json.load(st_json)
                if it == frame-1:
                    data_act.append(state['ax'])
                    data_act.append(state['omega'])
                else:
                    data.append(state['ax'])
                    data.append(state['omega'])
                data.append(state['v'])
                data.append(state['decision'])
                data.append(state['deviation'])
                n_objects = len(state['objects'])
                for i in range(MAX_N_OBJECTS):
                    if i < n_objects:
                        obj = state['objects'][i]
                        data.append(obj['x'])
                        data.append(obj['y'])
                        data.append(obj['theta'])
                        data.append(obj['v'])
                        data.append(obj['ax'])
                        data.append(obj['omega'])
                    else:
                        # add dummy
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
            rt.append(data)
            act.append(data_act)
            case.append(check_exp_case(data_index))
    return torch.FloatTensor(rt), torch.FloatTensor(act),torch.FloatTensor(case)

def load_negative_dataset(neg_path,frame):
    """
    return
    dataset : N * STATE_DIM
    N means # of data
    STATE_DIM means dimension of state (5 + N_OBJECTS * 6)
    """
    rt = []
    act = []
    case = []
    for data_index in negative_index:
        data_name = data_name_list[data_index]
        data_path = neg_path + data_name + "/"
        state_path = data_path + "state/"
        CASE_NUM = int(data_name.split('_')[3])
        st = seq_list[data_index][0]
        en = seq_list[data_index][1]
        for seq in range(st + 1, en + 1-frame):
            data = []
            data_act = []
            for it in range(frame):
                state_file = state_path + str(seq+it).zfill(6) + ".json"
                with open(state_file, "r") as st_json:
                    state = json.load(st_json)
                if it == frame-1:
                    data_act.append(state['ax'])
                    data_act.append(state['omega'])
                else:
                    data.append(state['ax'])
                    data.append(state['omega'])
                data.append(state['v'])
                data.append(state['decision'])
                data.append(state['deviation'])
                n_objects = len(state['objects'])
                for i in range(MAX_N_OBJECTS):
                    if i < n_objects:
                        obj = state['objects'][i]
                        data.append(obj['x'])
                        data.append(obj['y'])
                        data.append(obj['theta'])
                        data.append(obj['v'])
                        data.append(obj['ax'])
                        data.append(obj['omega'])
                    else:
                        # add dummy
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
                        data.append(0)
            rt.append(data)
            act.append(data_act)
            case.append(check_neg_case(data_index))
    
    return torch.FloatTensor(rt), torch.FloatTensor(act), torch.FloatTensor(case)

torch.manual_seed(0)

class MixQuality():
    def __init__(self,root = "./dataset/mixquality/",train=True,neg=False,norm=True,exp_case=[1,2,3],frame=1):
        exp_path = root + "exp/"
        neg_path = root + "neg/"
        self.train=train
        self.neg = neg
        self.e_in, self.e_target,self.e_case = load_expert_dataset(exp_path,exp_case,frame)
        self.n_in, self.n_target,self.n_case = load_negative_dataset(neg_path,frame)
        # print(self.e_in.size(),self.n_in.size(),self.e_target.size(),self.n_target.size(),frame)
        self.e_size = self.e_in.size(0)
        self.n_size = self.n_in.size(0)
        self.norm = norm
        self.mean_in = torch.mean(torch.cat((self.e_in,self.n_in),dim=0),dim=0)
        self.std_in = torch.std(torch.cat((self.e_in,self.n_in),dim=0),dim=0)
        self.mean_t = torch.mean(torch.cat((self.e_target,self.n_target),dim=0),dim=0)
        self.std_t = torch.std(torch.cat((self.e_target,self.n_target),dim=0),dim=0)

        self.load()
        self.normaize()

    def load(self):
        rand_e_idx = torch.randperm(self.e_size)
        # rand_n_idx = torch.randperm(self.n_size)
        if self.train:
            e_idx = rand_e_idx[:int(self.e_size*0.8)] # 8000
            # n_idx = rand_n_idx[:4500]
            e_in = self.e_in[e_idx]
            e_target = self.e_target[e_idx]
            # n_in = self.n_in[n_idx]
            # n_target = self.n_target[n_idx]
            # case = self.n_case[n_idx]
            self.e_label = e_idx.size(0)
            self.x = e_in
            self.y = e_target
            self.case = self.e_case[e_idx]
            # self.x = torch.cat((e_in,n_in),dim=0)
            # self.y = torch.cat((e_target,n_target),dim=0)
            # self.is_expert = torch.cat((torch.ones_like(e_idx),torch.zeros_like(n_idx)),dim=0)
            # self.case = torch.cat((torch.zeros_like(e_idx),case),dim=0)
        else:
            e_idx = rand_e_idx[int(self.e_size*0.8):]
            # n_idx = rand_n_idx[4500:]
            if not self.neg:
                self.x = self.e_in[e_idx]
                self.y = self.e_target[e_idx]
                self.case = self.e_case[e_idx]
                #self.is_expert = torch.ones_like(e_idx)
            else:
                self.x = self.n_in
                self.y = self.n_target
                self.case = self.n_case
                # self.x = self.n_in[n_idx]
                # self.y = self.n_target[n_idx]
                # self.case = self.n_case[n_idx]
                #self.is_expert = torch.zeros_like(n_idx)
            self.e_label = e_idx.size(0)
    def normaize(self):
        if self.norm:

            self.x = (self.x - self.mean_in)/(self.std_in)
            self.y = (self.y - self.mean_t)/(self.std_t)
            self.x[self.x != self.x] = 0
            self.y[self.y != self.y] = 0
        else:
            self.max_in = torch.max(torch.cat((self.e_in,self.n_in),dim=0),dim=0)[0]
            self.min_in = torch.min(torch.cat((self.e_in,self.n_in),dim=0),dim=0)[0]
            self.max_t = torch.max(torch.cat((self.e_target,self.n_target),dim=0),dim=0)[0]
            self.min_t = torch.min(torch.cat((self.e_target,self.n_target),dim=0),dim=0)[0]
            self.x = (self.x - self.min_in)/(self.max_in-self.min_in)
            self.y = (self.y - self.min_t)/(self.max_t-self.min_t)

if __name__ == '__main__':
    m = MixQuality(root='../dataset/mixquality/',train=False,neg=True)
    print(m.y[:100])
