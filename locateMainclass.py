import wntr
import pandas as pd
import matplotlib.pyplot as plt
from wntr.network.io import write_inpfile
import random
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
class LeakSimulator:
    def __init__(self,pm_nodes,leak_nodes,hdm_file):
        # leak model file
        self.leakModelFile = 'leakModel.inp'
        # hydraulic model file
        self.hdm_file = hdm_file
        # hydraulic model file update
        self.hdm_file_update = 'hydraulic_model_update.inp'
        # pressure measure node
        self.pm_nodes = pm_nodes # node measure
        # leak node
        self.leak_nodes = leak_nodes
        # Name Reservoir
        self.nameReservoir = self.getNameReservoir()
        # Name pipes measure
        self.namePipes_fm = self.getNameFlowMPIPES()
    def getNameFlowMPIPES(self):
        return [f'revtoj-{i+1}' for i in range(len(self.pm_nodes))]
    def getNameReservoir(self):
        return [f'rev-{i+1}' for i in range(len(self.pm_nodes))]
    def build_leak_model_from_hydraulic(self,hdm_file):
        pressure_pm_node = self.getPressureHym(hdm_file) # หาแรงดันที่จุดตรวจวัด
        wn = wntr.network.WaterNetworkModel(hdm_file) # เข้าถึงโครงข่ายแบบจำลอง hydraulic model
        for i, (pmn, fmp, nres) in enumerate(zip(self.pm_nodes, self.namePipes_fm, self.nameReservoir)):
            pattern_value = list(pressure_pm_node[pmn])
            pattern_name = f'head_pattern{i + 1}'
            head_pattern = wntr.network.elements.Pattern(pattern_name, pattern_value) # สร้าง head pattern
            wn.add_pattern(pattern_name, head_pattern) # เพิ่ม Head pattern ลงในโครงข่าย
            x,y = wn.nodes[pmn].coordinates # หาพิกัดของจุดตรวจวัดที่จะนำ Virtual Reservoir ไปต่อ
            # สร้าง Virtual Reservoir โดยกำหนดให้ total head = 1, นำ Head pattern จากที่หามาได้ใส่ใน Virtual Reservoir
            wn.add_reservoir(nres, base_head=1, head_pattern=pattern_name, coordinates=(x+1000, y+1000))
            wn.add_junction(f'J-rev-{i + 1}', base_demand=0, elevation=0, demand_pattern=None, coordinates=(x+500, y+500))
            # ขนาด,ความขรุขระของท่อและวาล์วถูกกำหนดเป็นค่าคงที่แทนการใช้ค่าจากท่อข้างๆ
            wn.add_pipe(fmp, f'rev-{i + 1}', f'J-rev-{i + 1}', length=100, diameter=0.1,
                        roughness=130, minor_loss=0)
            wn.add_valve(f"valve-rev{i + 1}", pmn, f'J-rev-{i + 1}', diameter=0.1, valve_type='PRV',
                         initial_status='OPEN')
        write_inpfile(wn, self.leakModelFile) # สร้าง leak model ไฟล์
    def getPressureHym(self, hdm_file): # แรงดันที่จุดตรวจวัดขณะไม่มีการรั่วไหล (hydraulic model)
        wn = wntr.network.WaterNetworkModel(hdm_file)
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        pressure_pm_node = round(results.node['pressure'][self.pm_nodes], 6)
        return pressure_pm_node
    def generate_sensitivity_matrix(self,leak_nodes,leaks,times,leakModel):
        output_dir_sensitivity = f"output{self.pm_nodes}/sensitivity"
        os.makedirs(output_dir_sensitivity, exist_ok=True)
        wn = wntr.network.WaterNetworkModel(leakModel)  # เชื่อมต่อกับ Leakage Model
        for leak_node in leak_nodes:
            dict_sensitive = {}
            delta = 0.1 # กำหนดระยะห่างของการหาความชัน sensitivity หรือ เดลต้า Demand
            gnLeak = wn.get_node(leak_node)
            base_demand = gnLeak.demand_timeseries_list[0].base_value * 3600 # อ่านค่า base demand
            # 1.) ดึงค่า Pattern จากโหนดผู้ใช้น้ำ ที่มีโอกาสเป็นจุดรั่ว (leak_nodes)
            # ตรวจสอบว่า node มี demand_timeseries และมี pattern กำหนดไว้หรือไม่
            if gnLeak.demand_timeseries_list and gnLeak.demand_timeseries_list[0].pattern:
                # ดึงชื่อ Pattern ที่ node ใช้
                pattern_name = gnLeak.demand_timeseries_list[0].pattern
                # ดึง pattern object
                pattern_obj = wn.get_pattern(pattern_name)
                # แปลง multipliers เป็น DataFrame
                pat_df = pd.DataFrame({
                    'Time Step': range(len(pattern_obj.multipliers)),
                    'Multiplier': pattern_obj.multipliers
                })
                print(f'Node {leak_node} มี Pattern การใช้น้ำเดิมชื่อ {pattern_name}')
            else:
                print(f'Node {leak_node} ไม่มี Pattern หรือไม่มีการตั้งค่า demand_timeseries_list')
                continue

            for leak in leaks:
                # 2.) หา sensitivity delta Pressure / delta leak flow rate(=delta)
                # 2.1) หา Pressure ที่จุดตรวจวัด เมื่อมีการรั่วปริมาณ Q+delta และ Q
                pressure1 = self.add_leakage_demand_to_pattern(wn,gnLeak,leak,leak_node,base_demand,pat_df,pattern_name,times,delta)
                pressure2 = self.add_leakage_demand_to_pattern(wn,gnLeak,leak,leak_node,base_demand,pat_df,pattern_name,times,0)
                # 2.2) หา sensitivity
                sensitivity = [(p2 - p1) / (delta) for (p2, p1) in zip(pressure2, pressure1)]
                # ทำ Sensitivity ให้เป็น Dataframe พร้อมนำไปแบ่ง section ต่อ
                dict_sensitive[leak] = sensitivity
                df_sensitive = pd.DataFrame.from_dict(dict_sensitive, orient='index', columns=self.nameReservoir) # แปลง dictionary เป็น Dataframe
                df_sensitive.index.name = 'Demand'
                df_sensitive = df_sensitive.T
            filename_sensitivity = os.path.join(output_dir_sensitivity, f"sensitivity_{leak_node}.csv")
            df_sensitive.to_csv(filename_sensitivity)
            print("บันทึก Sensitivity ลงไฟล์ CSV ในโฟลเดอร์ Output แล้ว")
    def generate_section_sensitivity_matrix(self,leak_nodes,leaks,times,leakModel):
        output_dir_section_sensitivity = f"output{self.pm_nodes}/section_sensitivity"
        os.makedirs(output_dir_section_sensitivity, exist_ok=True)
        wn = wntr.network.WaterNetworkModel(leakModel)  # เชื่อมต่อกับ Leakage Model
        for leak_node in leak_nodes:
            dict_sensitive = {}
            delta = 0.1 # กำหนดระยะห่างของการหาความชัน sensitivity หรือ เดลต้า Demand
            gnLeak = wn.get_node(leak_node)
            base_demand = gnLeak.demand_timeseries_list[0].base_value * 3600 # อ่านค่า base demand
            # 1.) ดึงค่า Pattern จากโหนดผู้ใช้น้ำ ที่มีโอกาสเป็นจุดรั่ว (leak_nodes)
            # ตรวจสอบว่า node มี demand_timeseries และมี pattern กำหนดไว้หรือไม่
            if gnLeak.demand_timeseries_list and gnLeak.demand_timeseries_list[0].pattern:
                # ดึงชื่อ Pattern ที่ node ใช้
                pattern_name = gnLeak.demand_timeseries_list[0].pattern
                # ดึง pattern object
                pattern_obj = wn.get_pattern(pattern_name)
                # แปลง multipliers เป็น DataFrame
                pat_df = pd.DataFrame({
                    'Time Step': range(len(pattern_obj.multipliers)),
                    'Multiplier': pattern_obj.multipliers
                })
                print(f'Node {leak_node} มี Pattern การใช้น้ำเดิมชื่อ {pattern_name}')
            else:
                print(f'Node {leak_node} ไม่มี Pattern หรือไม่มีการตั้งค่า demand_timeseries_list')
                continue

            for leak in leaks:
                # 2.) หา sensitivity delta Pressure / delta leak flow rate(=delta)
                # 2.1) หา Pressure ที่จุดตรวจวัด เมื่อมีการรั่วปริมาณ Q+delta และ Q
                pressure1 = self.add_leakage_demand_to_pattern(wn,gnLeak,leak,leak_node,base_demand,pat_df,pattern_name,times,delta)
                pressure2 = self.add_leakage_demand_to_pattern(wn,gnLeak,leak,leak_node,base_demand,pat_df,pattern_name,times,0)
                # 2.2) หา sensitivity
                sensitivity = [(p2 - p1) / (delta) for (p2, p1) in zip(pressure2, pressure1)]
                # ทำ Sensitivity ให้เป็น Dataframe พร้อมนำไปแบ่ง section ต่อ
                dict_sensitive[leak] = sensitivity
                df_sensitive = pd.DataFrame.from_dict(dict_sensitive, orient='index', columns=self.nameReservoir) # แปลง dictionary เป็น Dataframe
                df_sensitive.index.name = 'Demand'
                df_sensitive = df_sensitive.T

                # 3.) แบ่ง Section ของ sensitive ทุกๆช่วง
                n_column = df_sensitive.columns
                df_Section_sensitive = df_sensitive.copy()
                col = []
                for index in range(len(n_column) - 1):
                    df_Section_sensitive[f'{round(n_column[index], 1)}-{round(n_column[index + 1], 1)}'] = (df_sensitive[n_column[index]]+df_sensitive[n_column[index + 1]]) / 2
                    col.append(f'{round(n_column[index], 1)}-{round(n_column[index + 1], 1)}')
                df_Section_sensitive = df_Section_sensitive[col]
            filename_section_sensitivity = os.path.join(output_dir_section_sensitivity, f"section_sensitivity_{leak_node}.csv")
            df_Section_sensitive.to_csv(filename_section_sensitivity)
            print("บันทึก Sensitivity Section ลงไฟล์ CSV ในโฟลเดอร์ Output แล้ว")
    def plot_sensitivity(self):
        output_dir_csv = f"output{self.pm_nodes}/sensitivity"
        output_dir_graph = f"output{self.pm_nodes}/Graph_sensitivity"
        os.makedirs(output_dir_graph, exist_ok=True)
        # 1.) วนอ่านทุกไฟล์ sensitivity
        for filename in os.listdir(output_dir_csv):
            if filename.endswith(".csv"):
                file_path = os.path.join(output_dir_csv, filename)
                df = pd.read_csv(file_path, index_col=0)

                # 2.) กำหนดแกน x เป็น Demand (Leak Flow Rate) และ y เป็นค่า Sensitivity (ΔH / ΔQ)
                x = df.columns.astype(float)  # demand
                for reservoir_name, row in df.iterrows():
                    y = row.values
                    plt.plot(x, y, label=reservoir_name)

                # 3.) plot และ save ไฟล์ png ลงเครื่อง
                plt.title(f"Sensitivity vs Demand - {filename.replace('.csv', '')}")
                plt.xlabel("Demand (Leak Flow Rate)")
                plt.ylabel("Sensitivity (ΔP / ΔQ)")
                plt.legend(loc='best', fontsize=8)
                plt.grid(True)
                plt.tight_layout()
                save_path = os.path.join(output_dir_graph, filename.replace(".csv", ".png"))
                plt.savefig(save_path)
                plt.close()
                print(f"บันทึกกราฟ: {save_path}")
    def add_leakage_demand_to_pattern(self,wn,gnLeak,leak,leak_node,base_demand,pat_df,pattern_name,times,delta):
        leak_pat = pat_df.copy()
        leak_pat.loc[times] += (leak + delta) / base_demand
        pattern_leak_name = f'leak-p-{leak_node}+{delta}'
        wn.add_pattern(pattern_leak_name, leak_pat['Multiplier'].tolist())  # ใส่ pattern leak ลง network
        gnLeak.demand_timeseries_list[0]._pattern = pattern_leak_name  # ใส่ pattern leak ลง node
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        pressure = results.node['pressure'][self.pm_nodes].sum().tolist()
        pressure = [round(i, 6) for i in pressure]
        wn.remove_pattern(pattern_leak_name)  # remove pattern for reset
        gnLeak.demand_timeseries_list[0]._pattern = pattern_name
        return pressure
    def demandRandom(self,input_file,leak_nodes,output_file):
        output_dir_pattern = f"output{self.pm_nodes}/pattern"
        os.makedirs(output_dir_pattern, exist_ok=True)
        wn = wntr.network.WaterNetworkModel(input_file)
        for leak_node in leak_nodes:
            # 1.) ดึงค่า Pattern ใน node นั้น
            gnLeak = wn.get_node(leak_node)
            if gnLeak.demand_timeseries_list and gnLeak.demand_timeseries_list[0].pattern:
                # ดึงชื่อ Pattern ที่ node ใช้
                pattern_name = gnLeak.demand_timeseries_list[0].pattern
                # ดึง pattern object
                pattern_obj = wn.get_pattern(pattern_name)
                # แปลง multipliers เป็น DataFrame
                pat = pattern_obj.multipliers
            else:
                print(f'Node {leak_node} ไม่มี Pattern หรือไม่มีการตั้งค่า demand_timeseries_list')
                continue
            # 2.) สุ่ม Pattern
            pRand = [x * random.uniform(0.9, 1.1) for x in pat]
            avg = sum(pRand) / len(pRand)
            pRand = [x / avg for x in pRand] # ปรับให้ pattern มีค่าเฉลี่ย = 1
            # 3.) นำ pattern ออกเป็น csv
            index = [i+1 for i in range(len(pRand))]
            df = pd.DataFrame(pRand,columns=['Multiplier'],index=index)
            df.index.name = "Time"
            filename = os.path.join(output_dir_pattern,
                                                        f"pattern_{leak_node}.csv")
            df.to_csv(filename,index=True)
            # 4.) ใส่ pattern ลงใน node
            wn.add_pattern(f'p-{leak_node}', pRand)  # ค่า Pattern ตามช่วงเวลาคูณด้วย random
            junction = wn.get_node(leak_node)
            junction.demand_timeseries_list[0]._pattern = f'p-{leak_node}'
        write_inpfile(wn, output_file)
    def reportflowfromLeakage_timepat(self,leak_model,leak_nodes,times,leaks): # รายงาน flow ขณะเกิดการรั่วไหล
        output_dir_flow = f"output{self.pm_nodes}/reportFlow"
        os.makedirs(output_dir_flow, exist_ok=True)
        output_dir_flow_png = f'output{self.pm_nodes}/resultFlow_graph'
        os.makedirs(output_dir_flow_png, exist_ok=True)
        wn = wntr.network.WaterNetworkModel(leak_model)
        for leak_node in leak_nodes:
            print(f'เมื่อเกิดการรั่วที่ {leak_node}')
            print('กำลังประมวลผล...')
            filename_dir_flow = os.path.join(output_dir_flow, f"report-flow@{leak_node}.xlsx")
            with pd.ExcelWriter(filename_dir_flow) as writer:
                dataframes = []
                gnLeak = wn.get_node(leak_node)
                base_demand = gnLeak.demand_timeseries_list[0].base_value * 3600  # หน่วย CMH
                if gnLeak.demand_timeseries_list and gnLeak.demand_timeseries_list[0].pattern:
                    # ดึงชื่อ Pattern ที่ node ใช้
                    pattern_name = gnLeak.demand_timeseries_list[0].pattern
                    # ดึง pattern object
                    pattern_obj = wn.get_pattern(pattern_name)
                    # แปลง multipliers เป็น DataFrame
                    pat_df = pd.DataFrame({
                        'Time Step': range(len(pattern_obj.multipliers)),
                        'Multiplier': pattern_obj.multipliers
                    })
                    print(f'Node {leak_node} มี Pattern การใช้น้ำเดิมชื่อ {pattern_name}')
                else:
                    print(f'Node {leak_node} ไม่มี Pattern หรือไม่มีการตั้งค่า demand_timeseries_list')
                    continue

                for leak in leaks:
                    leak_pat = pat_df.copy()
                    leak_pat.loc[times] += leak/base_demand
                    pattern_leak_name = f'leak-p-{leak_node}'
                    wn.add_pattern(pattern_leak_name,leak_pat['Multiplier'].tolist()) # ใส่ pattern leak ลง network
                    gnLeak.demand_timeseries_list[0]._pattern = pattern_leak_name # ใส่ pattern leak ลง node
                    sim = wntr.sim.EpanetSimulator(wn)
                    results = sim.run_sim()
                    flow_pm_pipe = results.link['flowrate'][self.namePipes_fm] * 3600
                    flow_pm_pipe.index = flow_pm_pipe.index / 3600
                    dataframes.append(flow_pm_pipe)
                    # แสดงผลเป็นตาราง
                    flow_pm_pipe.to_excel(writer, sheet_name=f'leak{leak}CMH')
                    wn.remove_pattern(pattern_leak_name) # ลบ pattern leak
                    gnLeak.demand_timeseries_list[0]._pattern = pattern_name
                print(f'บันทึกตารางแสดงปริมาณการไหลรายชั่วโมงลงไฟล์ report-flow@{leak_node}.xlsx')
                fig, axes = plt.subplots(len(leaks), 1, figsize=(20, len(self.namePipes_fm) * 14), sharex=True)
                for i, (df, ax, leak) in enumerate(zip(dataframes, axes, leaks)):
                    for col in df.columns:
                        ax.plot(df.index, df[col], label=f"{col}")
                    ax.set_title(f"Flow for leakage {leak} CMH")
                    ax.set_ylabel("Flow Rate (m³/H)")
                    ax.legend()
                    ax.grid(True)
                # ตั้งชื่อแกน X
                plt.xlabel("Time")
                plt.tight_layout()
                filename_dir_flow_png = os.path.join(output_dir_flow_png, f"graph-flow-report@{leak_node}.png")
                plt.savefig(filename_dir_flow_png)
                plt.close()
                print(f'บันทึกกราฟแสดงปริมาณการไหลรายชั่วโมงลงไฟล์ graph-flow-report@{leak_node}.png')
    def rootmeansquare(self,sen_Leak,indicators,leak_nodes,piority): # หาจุดรั่วที่ได้จากการตรวจพบ 3 อันดับที่มี rms น้อยที่สุด
        rmss = []
        dict_mn = {}
        for indicator in indicators:
            df = (indicator - sen_Leak) ** 2
            s = df.sum()
            rms = s ** 0.5
            rmss.append(rms)
        #print(len(rmss))
        for (i, l) in enumerate(leak_nodes):
            dict_mn[l] = rmss[i]
        df_mn = pd.Series(dict_mn)
        return df_mn.nsmallest(piority).index.tolist()
    def cross_similarity(self,sen_Leak,indicators,assume_leak_nodes,piority):
        results = []
        for assume_leak_node, indicator in zip(assume_leak_nodes, indicators):
            sim = self.cosine_similarity(sen_Leak, indicator)
            results.append({
                'Node Detection Leak': assume_leak_node,
                'cos_sim': sim,
            })

        # จัดอันดับตาม cosine similarity มากที่สุด
        results_sorted = sorted(results, key=lambda x: x['cos_sim'], reverse=True)[:piority]
        return [res['Node Detection Leak'] for res in results_sorted]
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def evaluate_leak_detection_from_section_sensitivity(self,leakModel,db_sensitivity,assume_leak_nodes,assume_leaks,times): # sim ตรวจสอบการหาจุดรั่ว
        output_dir_result_sim = f"output{self.pm_nodes}/result_sim"
        os.makedirs(output_dir_result_sim, exist_ok=True)
        result_excel = os.path.join(output_dir_result_sim, f"ResultFromDetection(Section Sensitivity Matrix).xlsx")
        with pd.ExcelWriter(result_excel) as writer:
            # 1.) อ่าน sensitivity หรือ section_sensitivity จาก ไฟล์ .csv
            db = db_sensitivity
            # 2.) หา sensitivity ของจุดรั่ว
            result_dict_rms = {}
            result_dict_cs = {}
            dict_sensitive = {}
            aln = []
            al = []
            mn_rms = []
            mn_cs = []
            wn = wntr.network.WaterNetworkModel(leakModel)
            for assume_leak_node in assume_leak_nodes:
                print(f'รั่วที่ node {assume_leak_node}')
                delta = 0.1  # กำหนดระยะห่างของการหาความชัน sensitivity หรือ เดลต้า Demand
                gnLeak = wn.get_node(assume_leak_node)
                base_demand = gnLeak.demand_timeseries_list[0].base_value * 3600  # อ่านค่า base demand
                # 2.1) ดึงค่า Pattern จากโหนดผู้ใช้น้ำ ที่มีโอกาสเป็นจุดรั่ว (leak_nodes)
                # ตรวจสอบว่า node มี demand_timeseries และมี pattern กำหนดไว้หรือไม่
                if gnLeak.demand_timeseries_list and gnLeak.demand_timeseries_list[0].pattern:
                    # ดึงชื่อ Pattern ที่ node ใช้
                    pattern_name = gnLeak.demand_timeseries_list[0].pattern
                    # ดึง pattern object
                    pattern_obj = wn.get_pattern(pattern_name)
                    # แปลง multipliers เป็น DataFrame
                    pat_df = pd.DataFrame({
                        'Time Step': range(len(pattern_obj.multipliers)),
                        'Multiplier': pattern_obj.multipliers
                    })
                    print(f'Node {assume_leak_node} มี Pattern การใช้น้ำเดิมชื่อ {pattern_name}')
                else:
                    print(f'Node {assume_leak_node} ไม่มี Pattern หรือไม่มีการตั้งค่า demand_timeseries_list')
                    continue

                # 2.2) หา sensitivity delta Pressure / delta Leak flow rate(=delta)
                for assume_leak in assume_leaks:
                    # หา Pressure ที่จุดตรวจวัด เมื่อมีการรั่วปริมาณ Q+delta และ Q
                    head1 = self.add_leakage_demand_to_pattern(wn, gnLeak, assume_leak, assume_leak_node, base_demand, pat_df,
                                                               pattern_name, times, delta)
                    head2 = self.add_leakage_demand_to_pattern(wn, gnLeak, assume_leak, assume_leak_node, base_demand, pat_df,
                                                               pattern_name, times, 0)
                    # 2.3) หา sensitivity
                    sensitivity = [(h2 - h1) / (delta) for (h2, h1) in zip(head2, head1)]
                    # ทำ Sensitivity ให้เป็น Dataframe พร้อมนำไปแบ่ง section ต่อ
                    dict_sensitive[assume_leak] = sensitivity
                    df_sensitive = pd.DataFrame.from_dict(dict_sensitive, orient='index',
                                                          columns=self.nameReservoir)  # แปลง dictionary เป็น Dataframe
                    df_sensitive.index.name = 'Demand'
                    df_sensitive = df_sensitive.T
                    senLeak = df_sensitive[assume_leak]

                    # 3.) เปรียบเทียบ sensitivity ของจุดรั่วกับ Section Sensitivity Matrix หรือ Sensitivity Matrix
                    # 3.1) หาว่า assume leak ดังกล่าวอยู่ในช่วง section ไหน หรือ ตรงกับ sensitivity คอลัมป์ไหน
                    indicators = []  # คือ ตัวแปรที่เก็บ sensitivity ของช่วงที่ถูกเลือก
                    cols = db[0].columns.tolist()
                    for df_section_sensitive in db:
                        for col in cols:
                            lower, upper = map(float, col.split("-"))  # แยกช่วงตัวเลขจากชื่อคอลัมน์
                            if lower <= assume_leak <= upper:
                                target_column = col
                                indicators.append(df_section_sensitive[target_column])
                                break
                    aln.append(assume_leak_node)
                    al.append(assume_leak)

                    # 3.2) compare
                    # rms
                    min_node_rms = self.rootmeansquare(senLeak, indicators, assume_leak_nodes,
                                                     3)  # หาจุดรั่วที่ได้จากการตรวจพบ sequence อันดับที่มี rms น้อยที่สุด
                    mn_rms.append(min_node_rms)
                    # cross similarity
                    max_node_cs = self.cross_similarity(senLeak, indicators, assume_leak_nodes, 3)
                    mn_cs.append(max_node_cs)
            # 4.) บันทึกไฟล์ลง excel
            # rms
            result_dict_rms['assume leak node'] = aln
            result_dict_rms['assume leak'] = al
            result_dict_rms['Rank_1'] = [x[0] for x in mn_rms]
            result_dict_rms['Rank_2'] = [x[1] for x in mn_rms]
            result_dict_rms['Rank_3'] = [x[2] for x in mn_rms]
            df_result_rms = pd.DataFrame(result_dict_rms)
            df_result_rms['Check 1 rank'] = df_result_rms.apply(lambda row: row['assume leak node'] in row['Rank_1'],
                                                                axis=1)
            df_result_rms['Check 3 rank'] = df_result_rms.apply(lambda row: row['assume leak node'] in [row['Rank_1'], row['Rank_2'], row['Rank_3']],
                                                         axis=1)
            print(f'เช็ค 3 อันดับด้วย Root Mean Square')
            print(df_result_rms)
            df_result_rms.to_excel(writer, index=False, sheet_name="Result_rms")

            # cross similarity
            result_dict_cs['assume leak node'] = aln
            result_dict_cs['assume leak'] = al
            result_dict_cs['Rank_1'] = [x[0] for x in mn_cs]
            result_dict_cs['Rank_2'] = [x[1] for x in mn_cs]
            result_dict_cs['Rank_3'] = [x[2] for x in mn_cs]
            df_result_cs = pd.DataFrame(result_dict_cs)
            df_result_cs['Check 1 rank'] = df_result_cs.apply(lambda row: row['assume leak node'] in row['Rank_1'],
                                                              axis=1)
            df_result_cs['Check 3 rank'] = df_result_cs.apply(lambda row: row['assume leak node'] in [row['Rank_1'], row['Rank_2'], row['Rank_3']],
                                                       axis=1)
            print(f'เช็ค 3 อันดับด้วย Cross Similarity')
            print(df_result_cs)
            df_result_cs.to_excel(writer, index=False, sheet_name="Result_crossSim")

            # 5.) แสดง % ความผิดพลาด
            false_count_rms_3rank = (~df_result_rms['Check 3 rank']).sum()
            false_count_rms_1rank = (~df_result_rms['Check 1 rank']).sum()
            c_rms = len(df_result_rms)
            print(f'อัตราความแม่นยำของ Root Mean Square 3 อันดับแรก = {100-(false_count_rms_3rank / c_rms * 100) :.2f} % และอันดับแรก = {100-(false_count_rms_1rank/c_rms*100):.2f} %')

            false_count_cs_3rank = (~df_result_cs['Check 3 rank']).sum()
            false_count_cs_1rank = (~df_result_cs['Check 1 rank']).sum()
            c_cs = len(df_result_cs)
            print(f'อัตราความแม่นยำของ Cross Similarity 3 อันดับแรก = {100-(false_count_cs_3rank / c_cs * 100):.2f} % และอันดับแรก = {100-(false_count_cs_1rank/c_cs*100):.2f} %')
    def evaluate_leak_detection_from_sensitivity(self,leakModel,db_sensitivity,assume_leak_nodes,assume_leaks,times): # sim ตรวจสอบการหาจุดรั่ว
        output_dir_result_sim = f"output{self.pm_nodes}/result_sim"
        os.makedirs(output_dir_result_sim, exist_ok=True)
        result_excel = os.path.join(output_dir_result_sim, f"ResultFromDetection(Sensitivity Matrix).xlsx")
        with pd.ExcelWriter(result_excel) as writer:
            # 1.) อ่าน sensitivity หรือ section_sensitivity จาก ไฟล์ .csv
            db = db_sensitivity
            # 2.) หา sensitivity ของจุดรั่ว
            result_dict_rms = {}
            result_dict_cs = {}
            dict_sensitive = {}
            aln = []
            al = []
            mn_rms = []
            mn_cs = []
            wn = wntr.network.WaterNetworkModel(leakModel)
            for assume_leak_node in assume_leak_nodes:
                print(f'รั่วที่ node {assume_leak_node}')
                delta = 0.1  # กำหนดระยะห่างของการหาความชัน sensitivity หรือ เดลต้า Demand
                gnLeak = wn.get_node(assume_leak_node)
                base_demand = gnLeak.demand_timeseries_list[0].base_value * 3600  # อ่านค่า base demand
                # 2.1) ดึงค่า Pattern จากโหนดผู้ใช้น้ำ ที่มีโอกาสเป็นจุดรั่ว (leak_nodes)
                # ตรวจสอบว่า node มี demand_timeseries และมี pattern กำหนดไว้หรือไม่
                if gnLeak.demand_timeseries_list and gnLeak.demand_timeseries_list[0].pattern:
                    # ดึงชื่อ Pattern ที่ node ใช้
                    pattern_name = gnLeak.demand_timeseries_list[0].pattern
                    # ดึง pattern object
                    pattern_obj = wn.get_pattern(pattern_name)
                    # แปลง multipliers เป็น DataFrame
                    pat_df = pd.DataFrame({
                        'Time Step': range(len(pattern_obj.multipliers)),
                        'Multiplier': pattern_obj.multipliers
                    })
                    print(f'Node {assume_leak_node} มี Pattern การใช้น้ำเดิมชื่อ {pattern_name}')
                else:
                    print(f'Node {assume_leak_node} ไม่มี Pattern หรือไม่มีการตั้งค่า demand_timeseries_list')
                    continue

                # 2.2) หา sensitivity delta Pressure / delta Leak flow rate(=delta)
                for assume_leak in assume_leaks:
                    # หา Pressure ที่จุดตรวจวัด เมื่อมีการรั่วปริมาณ Q+delta และ Q
                    head1 = self.add_leakage_demand_to_pattern(wn, gnLeak, assume_leak, assume_leak_node, base_demand, pat_df,
                                                               pattern_name, times, delta)
                    head2 = self.add_leakage_demand_to_pattern(wn, gnLeak, assume_leak, assume_leak_node, base_demand, pat_df,
                                                               pattern_name, times, 0)
                    # 2.3) หา sensitivity
                    sensitivity = [(h2 - h1) / (delta) for (h2, h1) in zip(head2, head1)]
                    # ทำ Sensitivity ให้เป็น Dataframe พร้อมนำไปแบ่ง section ต่อ
                    dict_sensitive[assume_leak] = sensitivity
                    df_sensitive = pd.DataFrame.from_dict(dict_sensitive, orient='index',
                                                          columns=self.nameReservoir)  # แปลง dictionary เป็น Dataframe
                    df_sensitive.index.name = 'Demand'
                    df_sensitive = df_sensitive.T
                    senLeak = df_sensitive[assume_leak]

                    # 3.) เปรียบเทียบ sensitivity ของจุดรั่วกับ Section Sensitivity Matrix หรือ Sensitivity Matrix
                    # 3.1) หาว่า assume leak ดังกล่าวอยู่ในช่วง section ไหน หรือ ตรงกับ sensitivity คอลัมป์ไหน
                    indicators = []  # คือ ตัวแปรที่เก็บ sensitivity ของช่วงที่ถูกเลือก
                    cols = db[0].columns.tolist()
                    cols = [float(col) for col in cols]
                    for df_sensitiveMatrix in db:
                        # แปลง column ให้เป็น float ทั้งหมด
                        df_sensitiveMatrix.columns = df_sensitiveMatrix.columns.astype(float)
                        for col in cols:
                            if float(col) == float(assume_leak):
                                indicators.append(df_sensitiveMatrix[col])
                                break
                    aln.append(assume_leak_node)
                    al.append(assume_leak)

                    # 3.2) compare
                    # rms
                    min_node_rms = self.rootmeansquare(senLeak, indicators, assume_leak_nodes,
                                                     3)  # หาจุดรั่วที่ได้จากการตรวจพบ sequence อันดับที่มี rms น้อยที่สุด
                    mn_rms.append(min_node_rms)
                    # cross similarity
                    max_node_cs = self.cross_similarity(senLeak, indicators, assume_leak_nodes, 3)
                    mn_cs.append(max_node_cs)
            # 4.) บันทึกไฟล์ลง excel
            # rms
            result_dict_rms['assume leak node'] = aln
            result_dict_rms['assume leak'] = al
            result_dict_rms['Rank_1'] = [x[0] for x in mn_rms]
            result_dict_rms['Rank_2'] = [x[1] for x in mn_rms]
            result_dict_rms['Rank_3'] = [x[2] for x in mn_rms]
            df_result_rms = pd.DataFrame(result_dict_rms)
            df_result_rms['Check 1 rank'] = df_result_rms.apply(lambda row: row['assume leak node'] in row['Rank_1'],
                                                                axis=1)
            df_result_rms['Check 3 rank'] = df_result_rms.apply(lambda row: row['assume leak node'] in [row['Rank_1'], row['Rank_2'], row['Rank_3']],
                                                         axis=1)
            print(f'เช็ค 3 อันดับด้วย Root Mean Square')
            print(df_result_rms)
            df_result_rms.to_excel(writer, index=False, sheet_name="Result_rms")

            # cross similarity
            result_dict_cs['assume leak node'] = aln
            result_dict_cs['assume leak'] = al
            result_dict_cs['Rank_1'] = [x[0] for x in mn_cs]
            result_dict_cs['Rank_2'] = [x[1] for x in mn_cs]
            result_dict_cs['Rank_3'] = [x[2] for x in mn_cs]
            df_result_cs = pd.DataFrame(result_dict_cs)
            df_result_cs['Check 1 rank'] = df_result_cs.apply(lambda row: row['assume leak node'] in row['Rank_1'],
                                                              axis=1)
            df_result_cs['Check 3 rank'] = df_result_cs.apply(lambda row: row['assume leak node'] in [row['Rank_1'], row['Rank_2'], row['Rank_3']],
                                                       axis=1)
            print(f'เช็ค 3 อันดับด้วย Cross Similarity')
            print(df_result_cs)
            df_result_cs.to_excel(writer, index=False, sheet_name="Result_crossSim")

            # 5.) แสดง % ความผิดพลาด
            false_count_rms_3rank = (~df_result_rms['Check 3 rank']).sum()
            false_count_rms_1rank = (~df_result_rms['Check 1 rank']).sum()
            c_rms = len(df_result_rms)
            print(f'อัตราความแม่นยำของ Root Mean Square 3 อันดับแรก = {100-(false_count_rms_3rank / c_rms * 100) :.2f} % และอันดับแรก = {100-(false_count_rms_1rank/c_rms*100):.2f} %')

            false_count_cs_3rank = (~df_result_cs['Check 3 rank']).sum()
            false_count_cs_1rank = (~df_result_cs['Check 1 rank']).sum()
            c_cs = len(df_result_cs)
            print(f'อัตราความแม่นยำของ Cross Similarity 3 อันดับแรก = {100-(false_count_cs_3rank / c_cs * 100):.2f} % และอันดับแรก = {100-(false_count_cs_1rank/c_cs*100):.2f} %')
    def get_section_sensitivity(self): # ดึงค่า Section Sensitivity จาก csv
        output_dir_section_sensitivity = f"output{self.pm_nodes}/section_sensitivity"
        df_section_sensitives = []
        for filename in os.listdir(output_dir_section_sensitivity):
            if filename.endswith(".csv"):
                file_path = os.path.join(output_dir_section_sensitivity, filename)
                df = pd.read_csv(file_path, index_col=0)
                df_section_sensitives.append(df)
        return df_section_sensitives
    def get_sensitivity(self): # ดึงค่า Sensitivity จาก csv
        output_dir_sensitivity = f"output{self.pm_nodes}/sensitivity"
        df_sensitives = []
        for filename in os.listdir(output_dir_sensitivity):
            if filename.endswith(".csv"):
                file_path = os.path.join(output_dir_sensitivity, filename)
                df = pd.read_csv(file_path, index_col=0)
                df_sensitives.append(df)
        return df_sensitives