import os
import json
import argparse
from creat_graph import creat_metagraph
from camel.storages import Neo4jGraph

def list_checkpoints():
    """列出所有可用的检查点"""
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print("没有找到检查点目录")
        return
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
    
    if not checkpoints:
        print("没有找到检查点文件")
        return
    
    print("可用的检查点:")
    for i, checkpoint_file in enumerate(checkpoints, 1):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                gid = data.get("gid", "未知")
                current = data.get("current_index", 0)
                total = data.get("total_chunks", 0)
                processed = len(data.get("processed_chunks", []))
                print(f"{i}. GID: {gid} - 进度: {processed}/{total} ({processed/total*100:.1f}%)")
        except Exception as e:
            print(f"{i}. {checkpoint_file} - 无法读取: {e}")

def resume_processing(gid):
    """从检查点恢复处理
    Args:
        gid: 要恢复的图元素组ID
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{gid}.json")
    chunks_cache_path = os.path.join(checkpoint_dir, f"chunks_{gid}.json")
    
    if not os.path.exists(checkpoint_path):
        print(f"找不到GID为{gid}的检查点文件")
        return
    
    if not os.path.exists(chunks_cache_path):
        print(f"找不到GID为{gid}的分块缓存文件")
        return
    
    # 加载检查点和分块缓存
    try:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        
        with open(chunks_cache_path, "r") as f:
            chunks_data = json.load(f)
            
        # 设置Neo4j连接
        url = os.getenv("NEO4J_URL")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([url, username, password]):
            print("请设置Neo4j环境变量: NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD")
            return
        
        n4j = Neo4jGraph(
            url=url,
            username=username,
            password=password
        )
        
        # 创建一个模拟的args对象
        class Args:
            pass
        
        args = Args()
        args.grained_chunk = True  # 假设使用细粒度分块
        args.ingraphmerge = True  # 假设需要合并相似节点
        
        # 从缓存中获取原始内容
        # 这里我们需要从run.py中获取原始内容，但由于我们没有保存原始内容
        # 我们可以从数据库中查询该GID的Summary节点来获取
        query = """
        MATCH (s:Summary {gid: $gid})
        RETURN s.content as content
        LIMIT 1
        """
        result = n4j.query(query, {"gid": gid})
        
        if not result:
            print(f"无法从数据库中获取GID为{gid}的内容摘要")
            return
        
        content = result[0].get("content", "")
        if not content:
            print(f"GID为{gid}的内容摘要为空")
            return
        
        print(f"正在恢复GID为{gid}的处理...")
        creat_metagraph(args, content, gid, n4j)
        print(f"GID为{gid}的处理已恢复并完成")
        
    except Exception as e:
        print(f"恢复处理时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="从检查点恢复图处理")
    parser.add_argument("-list", action="store_true", help="列出所有可用的检查点")
    parser.add_argument("-resume", type=str, help="要恢复的图元素组ID")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.resume:
        resume_processing(args.resume)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()