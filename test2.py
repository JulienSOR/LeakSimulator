from locateMainclass import LeakSimulator

# input data
pm_nodes = ['J-6','J-7','J-8'] # จุดตรวจวัดแรงดัน หรือจุดรับรู้แรงดัน
leak_nodes = ['J-3','J-4','J-5','J-6','J-7','J-8','J-9'] # จุดรั่วทดสอบ
hdm_file = 'hydraulic_model.inp'

# 1. หา Sensitivity Matrix
# ข้อมูลการรั่วไหลสำหรับสร้าง Sensitivity Matrix
leaks = [0.1*i for i in list(range(0,81,1))] # CMH เอาไว้สร้างตาราง Sensitivity Matrix
times = list(range(15,18,1)) # leak ช่วงเวลาใช้น้ำน้อย 15:00 - 17:00 PM
#times = list(range(8,11,1)) # leak ช่วงเวลาใช้น้ำมาก 08:00 - 10:00 PM
print('ส่วนที่ 1 หา Sensitivity Matrix ')
# ส่วนการรัน
ll = LeakSimulator(pm_nodes,leak_nodes,hdm_file) # อ้างอิง Main class
ll.build_leak_model_from_hydraulic(hdm_file) # สร้าง leak model
leakModel = ll.leakModelFile # เข้าถึงชื่อไฟล์แบบจำลองรับรู้การรั่วไหล leakModel.inp
ll.generate_sensitivity_matrix(leak_nodes,leaks,times,leakModel) # สร้าง Sensitivity Matrix
ll.plot_sensitivity() # พล็อตความสัมพันธ์ระหว่าง Sensitivity กับ อัตราการรั่วไหล Leak Flow Rate (CMH)
print('*'*150)

# 2. ตรวจสอบการตรวจพบจุดรั่วไหล
print('ส่วนที่ 2 ตรวจสอบการตรวจพบจุดรั่วไหล ')
# ข้อมูลการรั่วไหลสำหรับการตรวจสอบการตรวจจับจุดรั่ว
assume_leak_nodes = leak_nodes # สมมุติจุดรั่ว
assume_leaks = [0.1*i for i in list(range(1,81,1))] # CMH
# ส่วนการรัน
db_sensitivity = ll.get_sensitivity() # ดึงข้อมูล Sensitivity Matrix ที่เก็บไว้ใน Data base (ไฟล์ CSV)
ll.evaluate_leak_detection_from_sensitivity(leakModel
                                            ,db_sensitivity
                                            ,assume_leak_nodes
                                            ,assume_leaks,times) # ตรวจสอบการตรวจจับจุดรั่วของแบบจำลอง