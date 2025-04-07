from locateMainclass import LeakSimulator

pm_nodes = ['J-6','J-7','J-8']  # ตำแหน่งจุดที่ใช้สังเกตแรงดัน (Pressure Monitor Nodes)
leak_nodes = ['J-3','J-4','J-5','J-6','J-7','J-8','J-9']  # จุดที่ต้องการทดสอบการรั่ว
hdm_file = "hydraulic_model.inp"  # ไฟล์แบบจำลองเครือข่าย

ll = LeakSimulator(pm_nodes,leak_nodes,hdm_file) # อ้างอิง Main class
ll.build_leak_model_from_hydraulic(hdm_file) # สร้าง leak model

# ข้อมูลการรั่วไหลสำหรับสร้าง Sensitivity Matrix
leaks = [0.1*i for i in list(range(0,81,1))] # CMH เอาไว้สร้างตาราง Sensitivity Matrix
times = list(range(8,11,1)) # leak 08:00 - 10:00 PM
leakModel = ll.leakModelFile
ll.generate_sensitivity_matrix(leak_nodes,leaks,times,leakModel) # สร้าง Sensitivity Matrix

ll.plot_sensitivity() # พล็อตความสัมพันธ์ระหว่าง Sensitivity กับ อัตราการรั่วไหล Leak Flow Rate (CMH)

db_sensitivity = ll.get_sensitivity() # ดุึงข้อมูล Sensitivity Matrix จากไฟล์ CSV มาใส่ในตัวแปร
assume_leak_nodes = leak_nodes # สมมุติจุดรั่ว
assume_leaks = [0.1*i for i in list(range(1,81,1))] # CMH
ll.evaluate_leak_detection_from_sensitivity(leakModel
                                            ,db_sensitivity
                                            ,assume_leak_nodes
                                            ,assume_leaks,times)

