from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def clean_graph(self):
        with self.driver.session() as session:
            session.write_transaction(self._delete_all)

    @staticmethod
    def _delete_all(tx):
        tx.run("MATCH (n {gid: '44a8569a-fec5-4686-bf21-7b34dc1f8140'}) DETACH DELETE n")

# Example usage
conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "zcx1264521752")
conn.clean_graph()
conn.close()
