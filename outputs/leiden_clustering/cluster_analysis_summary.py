import pandas as pd
import re

# 1. Load Current Data (for Total Queries & Unique Queries)
try:
    current_df = pd.read_csv('/home/ubuntu/Kshitij/KCC Analysis/outputs/leiden_clustering/global_summary.csv')
    print(f"Loaded {len(current_df)} rows from global_summary.csv.")
    print("Example Row:", current_df.iloc[-1].to_dict())  # Print last row for sanity check
    
    # Create lookup map: Normalized (lower+stripped) -> {total, unique}
    crop_stats = {}
    for idx, row in current_df.iterrows():
        name = str(row['crop'])
        norm_name = name.strip().lower()
        crop_stats[norm_name] = {
            'total_queries': row['total_queries'],
            'unique_queries': row['unique_queries']
        }
    print(f"Created map with {len(crop_stats)} entries")
except Exception as e:
    print(f"Error loading current summary: {e}")
    crop_stats = {}

# 2. Parse User Logs
log_text = """
[112/272] Melon: 133 clusters (7 cover 85%)
[113/272] Pig: 79 clusters (5 cover 85%)
[114/272] African Sarson: 110 clusters (8 cover 85%)
[115/272] Gladiolus: 265 clusters (86 cover 85%)
[116/272] Beet Root Garden BeetStock Beet: 179 clusters (13 cover 85%)
[117/272] Ber: 190 clusters (18 cover 85%)
[118/272] Sugar Beet: 212 clusters (33 cover 85%)
[119/272] Khesari chickling vetch grass pea: 162 clusters (27 cover 85%)
[120/272] Jamun: 116 clusters (7 cover 85%)
[121/272] Barnyard Millet KundiraivlliSawan: 348 clusters (179 cover 85%)
[122/272] fodder maize : 127 clusters (8 cover 85%)
[123/272] OPIUM POPY: 321 clusters (148 cover 85%)
[124/272] Castor Rehri Rendi Arandi : 342 clusters (200 cover 85%)
[125/272] Sesame (Gingelly/Til)/Sesamum: 378 clusters (245 cover 85%)
[126/272] POULTRY  FARM: 16 clusters (5 cover 85%)
[127/272] Sandal Wood: 110 clusters (7 cover 85%)
[128/272] Rajma french bean: 103 clusters (7 cover 85%)
[129/272] Mosambi: 115 clusters (8 cover 85%)
[130/272] Curry Leaf: 209 clusters (79 cover 85%)
[131/272] Orange: 116 clusters (9 cover 85%)
[132/272] Teak: 148 clusters (24 cover 85%)
[133/272] Horse Gram kulthikultha: 387 clusters (266 cover 85%)
[134/272] Cluster Bean: 162 clusters (37 cover 85%)
[135/272] ARUM : 240 clusters (119 cover 85%)
[136/272] Broccoli: 185 clusters (65 cover 85%)
[137/272] Sudan Grass: 91 clusters (11 cover 85%)
[138/272] Fennel: 116 clusters (12 cover 85%)
[139/272] Stevia: 133 clusters (23 cover 85%)
[140/272] Peach: 84 clusters (4 cover 85%)
[141/272] AmaranthusGrain Amaranthus: 174 clusters (65 cover 85%)
[142/272] Cold Water: 79 clusters (6 cover 85%)
[143/272] Bovine(Cow,Buffalo): 120 clusters (15 cover 85%)
[144/272] Soybean (bhat): 483 clusters (377 cover 85%)
[145/272] Napier Grass: 116 clusters (16 cover 85%)
[146/272] Broad Bean: 201 clusters (98 cover 85%)
[147/272] Strawberry: 84 clusters (6 cover 85%)
[148/272] Little Millet SamaiKutkikodo-kutki: 174 clusters (79 cover 85%)
[149/272] Peas (field peas/ garden peas/matar): 307 clusters (219 cover 85%)
[150/272] Brinjal: 22968 clusters (12986 cover 85%)
[151/272] Hibiscus Gurhal: 70 clusters (4 cover 85%)
[152/272] Sapota: 65 clusters (5 cover 85%)
[153/272] Lab Lab: 96 clusters (12 cover 85%)
[154/272] Dog: 82 clusters (9 cover 85%)
[155/272] Chapan Kaddu: 118 clusters (28 cover 85%)
[156/272] Duck: 76 clusters (6 cover 85%)
[157/272] Italian Millet ThenaiNavaneFoxtail MilletKang: 244 clusters (160 cover 85%)
[158/272] Cumin: 88 clusters (10 cover 85%)
[159/272] Chestnut: 121 clusters (46 cover 85%)
[160/272] Karan Rai: 120 clusters (41 cover 85%)
[161/272] Garden Pea: 90 clusters (17 cover 85%)
[162/272] Indian Clover Senji Sweet Clover: 142 clusters (73 cover 85%)
[163/272] White Yam: 101 clusters (19 cover 85%)
[164/272] Butter Pea Vegetable: 155 clusters (76 cover 85%)
[165/272] Methi Fenugreek: 52 clusters (6 cover 85%)
[166/272] Snake Gourd: 118 clusters (47 cover 85%)
[167/272] Safflower kusumkardi: 96 clusters (29 cover 85%)
[168/272] Koronda: 87 clusters (19 cover 85%)
[169/272] Kodo Millet KodaraVaragu: 165 clusters (93 cover 85%)
[170/272] Tapioca Cassava: 135 clusters (71 cover 85%)
[171/272] Long Melon: 80 clusters (15 cover 85%)
[172/272] Cardamom: 95 clusters (32 cover 85%)
[173/272] Colocasia (Arvi, Arbi): 76 clusters (10 cover 85%)
[174/272] Apple: 104 clusters (41 cover 85%)
[175/272] Brackish: 39 clusters (3 cover 85%)
[176/272] Lethyrus: 95 clusters (33 cover 85%)
[177/272] Jojoba: 215 clusters (151 cover 85%)
[178/272] Flying Duck: 55 clusters (8 cover 85%)
[179/272] Brussils Sprouts: 160 clusters (101 cover 85%)
[180/272] Niger Ramtil: 111 clusters (55 cover 85%)
[181/272] Kundru: 55 clusters (6 cover 85%)
[182/272] Chrysanthemum: 145 clusters (91 cover 85%)
[183/272] Almond: 103 clusters (45 cover 85%)
[184/272] Coconut: 62 clusters (11 cover 85%)
[185/272] Baby Corn: 141 clusters (87 cover 85%)
[186/272] Babool: 117 clusters (60 cover 85%)
[187/272] Olive: 208 clusters (157 cover 85%)
[188/272] Buffel Grass Anjan Grass: 82 clusters (29 cover 85%)
[189/272] Dolichos Bean: 98 clusters (49 cover 85%)
[190/272] Bell Pepper: 132 clusters (85 cover 85%)
[191/272] Bush Squash: 98 clusters (49 cover 85%)
[192/272] China Astor: 129 clusters (86 cover 85%)
[193/272] Custard Apple: 99 clusters (53 cover 85%)
[194/272] Leafy Vegetable: 114 clusters (70 cover 85%)
[195/272] Cotton (Kapas): 138 clusters (95 cover 85%)
[196/272] Kiwi Fruit: 77 clusters (34 cover 85%)
[197/272] Mulberry: 60 clusters (17 cover 85%)
[198/272] Fig: 56 clusters (14 cover 85%)
[199/272] Spine Gourd: 106 clusters (58 cover 85%)
[200/272] Jute: 117 clusters (76 cover 85%)
[201/272] Sal Wood: 111 clusters (71 cover 85%)
[202/272] Tumba: 63 clusters (22 cover 85%)
[203/272] Ivy Gourd: 77 clusters (37 cover 85%)
[204/272] Chinese Cabbage: 119 clusters (82 cover 85%)
[205/272] Jasmine: 85 clusters (46 cover 85%)
[206/272] Periwinkle: 109 clusters (69 cover 85%)
[207/272] Turnip Saljam: 77 clusters (42 cover 85%)
[208/272] Greater Yam: 100 clusters (65 cover 85%)
[209/272] Betel Vine: 93 clusters (54 cover 85%)
[210/272] Gerbera: 104 clusters (69 cover 85%)
[211/272] Moth Bean kidney bean deww gram: 113 clusters (77 cover 85%)
[212/272] Carnation: 108 clusters (73 cover 85%)
[213/272] Smooth Guard: 92 clusters (58 cover 85%)
[214/272] Turkey: 40 clusters (8 cover 85%)
[215/272] Caleus: 106 clusters (70 cover 85%)
[216/272] Indian Squash TindaRound Melon: 61 clusters (27 cover 85%)
[217/272] Honey plant: 99 clusters (64 cover 85%)
[218/272] Indian Bean Vegetable: 92 clusters (60 cover 85%)
[219/272] Apricot: 67 clusters (36 cover 85%)
[220/272] Drum Stick: 79 clusters (48 cover 85%)
[221/272] Pear: 41 clusters (12 cover 85%)
[222/272] Rajmash Bean: 58 clusters (24 cover 85%)
[223/272] Clove: 64 clusters (34 cover 85%)
[224/272] Pepper: 67 clusters (38 cover 85%)
[225/272] Spinach (Palak): 43 clusters (15 cover 85%)
[226/272] Plum: 70 clusters (42 cover 85%)
[227/272] Barley (Jau): 26 clusters (4 cover 85%)
[228/272] Pineapple: 46 clusters (17 cover 85%)
[229/272] Jatropha Ratanjot: 65 clusters (40 cover 85%)
[230/272] Loquat: 52 clusters (28 cover 85%)
[231/272] Avacado: 73 clusters (48 cover 85%)
[232/272] Raya (Indian Mustard): 32 clusters (9 cover 85%)
[233/272] Cymbidium: 86 clusters (63 cover 85%)
[234/272] Summer Squash Vegetable Marrow: 79 clusters (56 cover 85%)
[235/272] Mochai lab-lab: 76 clusters (51 cover 85%)
[236/272] Ash Gourd (Petha): 18 clusters (3 cover 85%)
[237/272] Crossandra: 74 clusters (49 cover 85%)
[238/272] Winged Bean: 71 clusters (48 cover 85%)
[239/272] Anthurium: 100 clusters (79 cover 85%)
[240/272] Walnut: 75 clusters (54 cover 85%)
[241/272] Guineafowl: 37 clusters (15 cover 85%)
[242/272] Celery: 60 clusters (41 cover 85%)
[243/272] Ribbed Gourd (Kali Tori): 16 clusters (4 cover 85%)
[244/272] Rabbit: 37 clusters (16 cover 85%)
[245/272] Goose: 46 clusters (26 cover 85%)
[246/272] Roselle Mesta: 65 clusters (46 cover 85%)
[247/272] Camel: 51 clusters (32 cover 85%)
[248/272] Chinar Tree: 72 clusters (54 cover 85%)
[249/272] Yard Long Bean: 66 clusters (48 cover 85%)
[250/272] Sheep: 40 clusters (21 cover 85%)
[251/272] Palmyra: 82 clusters (63 cover 85%)
[252/272] Knol-Khol: 71 clusters (53 cover 85%)
[253/272] Cinnamon: 75 clusters (58 cover 85%)
[254/272] Lilliums: 74 clusters (57 cover 85%)
[255/272] DatePalm: 59 clusters (42 cover 85%)
[256/272] Indian rapeseed and mustard (yellow sarson): 25 clusters (7 cover 85%)
[257/272] Horse: 53 clusters (37 cover 85%)
[258/272] Hazlenut: 54 clusters (38 cover 85%)
[259/272] Arecanut: 70 clusters (54 cover 85%)
[260/272] Tuberose: 55 clusters (40 cover 85%)
[261/272] Maize Makka: 32349 clusters (20891 cover 85%)
[262/272] Mentha: 25018 clusters (12659 cover 85%)
[263/272] Black Gram urd bean: 31843 clusters (19261 cover 85%)
[264/272] Chillies: 21071 clusters (8339 cover 85%)
[265/272] Paddy (Dhan): 59092 clusters (41531 cover 85%)
[266/272] Mango: 24685 clusters (6933 cover 85%)
[267/272] Mustard: 45407 clusters (26242 cover 85%)
[268/272] Potato: 74557 clusters (44435 cover 85%)
[269/272] Sugarcane Noble Cane: 117326 clusters (74901 cover 85%)
"""

results = []
# Adjusted regex to handle potential formatting variations
pattern = re.compile(r"\[\d+/\d+\] (.*?): (\d+) clusters \((\d+) cover 85%\)")

for line in log_text.strip().split('\n'):
    match = pattern.search(line)
    if match:
        crop_name = match.group(1).strip()
        total_clusters = int(match.group(2))
        clusters_85 = int(match.group(3))
        
        # Simple normalization: lower() and strip()
        norm_name = crop_name.lower().strip()
        
        # Look up in crop_stats
        stats = crop_stats.get(norm_name, {'total_queries': 0, 'unique_queries': 0})
        
        # If still 0, try fuzzier matching just in case (e.g. partial string)
        if stats['total_queries'] == 0:
            for k, v in crop_stats.items():
                if k in norm_name or norm_name in k: # Substring match
                    stats = v
                    break
                    
        results.append({
            'crop': crop_name,
            'total_queries': stats.get('total_queries', 0),
            'unique_queries': stats.get('unique_queries', 0),
            'total_clusters': total_clusters,
            'clusters_for_85pct': clusters_85
        })

# Save to CSV
output_df = pd.DataFrame(results)
output_path = '/home/ubuntu/Kshitij/KCC Analysis/outputs/leiden_clustering/leiden_cluster_analysis_previous_run.csv'
output_df.to_csv(output_path, index=False)
print(f"Saved {len(results)} rows to {output_path}")
print(output_df.head().to_string())
