import os
import torch
import time
import threading
from torch.cuda import device_count

# 设置可见的GPU（0-7号卡）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def gpu_worker(gpu_id, target_mb=10000, target_utilization=50, duration=60):
    """单个GPU的负载函数（每个线程负责一张卡）"""
    try:
        # 指定当前线程使用的GPU
        device = torch.device(f'cuda:{gpu_id}') 
        torch.cuda.set_device(device)
        
        print(f"GPU {gpu_id} 开始初始化...")
        if not torch.cuda.is_available():
            print(f"GPU {gpu_id} 不可用，跳过")
            return
        
        # 转换目标显存为字节（调整计算方式，更准确）
        target_bytes = target_mb * 1024 * 1024  # 1MB = 1024*1024字节
        
        # 计算需要分配的float32元素数量（每个元素4字节）
        element_size = 4
        num_elements = target_bytes // element_size
        if num_elements <= 0:
            print(f"GPU {gpu_id} 目标显存过小，无法分配")
            return
        
        # 分配显存（在当前GPU上）
        print(f"GPU {gpu_id} 尝试分配约 {target_mb}MB 显存...")
        data = torch.randn(num_elements, dtype=torch.float32, device=device)
        
        # 验证实际分配的显存
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"GPU {gpu_id} 成功分配: {allocated:.2f}MB 显存")
        
        # 计算负载参数（控制使用率）
        utilization = max(0, min(100, target_utilization)) / 100.0
        sleep_time = (1 - utilization) * 0.1  # 空闲时间比例
        
        # 计算密集型操作（消耗GPU算力）
        def compute_load():
            for _ in range(100):  # 计算强度，可根据需要调整
                data.mul_(1.0001).div_(1.0001)  # 简单但持续的计算
                # 强制同步，确保计算被执行
                torch.cuda.synchronize(device)
        
        print(f"GPU {gpu_id} 开始运行，目标使用率: {target_utilization}%")
        start_time = time.time()
        last_print = start_time
        
        while time.time() - start_time < duration:
            compute_load()  # 执行计算
            if sleep_time > 0:
                time.sleep(sleep_time)  # 控制负载率
            
            # 每隔5秒打印一次状态
            if time.time() - last_print > 5:
                last_print = time.time()
                elapsed = last_print - start_time
                print(f"GPU {gpu_id} 运行中... 已运行 {elapsed:.1f}秒 (剩余 {duration - elapsed:.1f}秒)")
        
        print(f"GPU {gpu_id} 运行完成，释放显存")
        
    except RuntimeError as e:
        print(f"GPU {gpu_id} 错误: {str(e)}")
    except KeyboardInterrupt:
        print(f"\nGPU {gpu_id} 被用户中断")
    finally:
        # 清理当前GPU的显存
        torch.cuda.empty_cache()
        print(f"GPU {gpu_id} 显存已释放")

def run_multi_gpu():
    # 参数设置（所有GPU共用的参数）
    MEMORY_MB = 30000    # 每张卡分配的显存大小(MB)
    UTILIZATION = 60    # 每张卡的目标使用率(0-100)
    DURATION = 100000000 # 总运行时间(秒)
    
    # 获取可见的GPU数量（根据CUDA_VISIBLE_DEVICES）
    num_gpus = device_count()
    print(f"检测到 {num_gpus} 张可用GPU，准备启动负载...")
    
    # 为每张卡创建一个线程
    threads = []
    for gpu_id in range(num_gpus):
        thread = threading.Thread(
            target=gpu_worker,
            args=(gpu_id,),
            kwargs={
                'target_mb': MEMORY_MB,
                'target_utilization': UTILIZATION,
                'duration': DURATION
            }
        )
        threads.append(thread)
        thread.start()
        time.sleep(1)  # 错开启动时间，避免同时分配显存导致冲突
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    run_multi_gpu()