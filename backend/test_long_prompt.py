import unittest
import json
import os
import threading
import time
import uuid
import tempfile
import shutil
from recursive.engine import GraphRunEngine, RegularDummyNode, NodeType, TaskStatus
# Graph might not be directly needed for import if engine handles its usage internally
# from recursive.graph import Graph

# Define a temporary directory for test outputs
RESULTS_DIR_NAME = "test_results_temp"
RECORDS_DIR_NAME = "records"

# Basic config for the dummy node and engine
TEST_CONFIG = {
    "language": "en",
    "action_mapping": {
        "plan": ["UpdateAtomPlanningAgent", {}], # Dummy agent
        # Add other actions if needed by the specific test flow
    },
    "task_type2tag": {
        "COMPOSITION": "write",
        "REASONING": "think",
        "RETRIEVAL": "search",
    },
    "tag2task_type": { # Inverse mapping
        "write": "COMPOSITION",
        "think": "REASONING",
        "search": "RETRIEVAL",
    },
    "require_keys": {
        "COMPOSITION": ["id", "dependency", "goal", "task_type", "length"],
        "RETRIEVAL": ["id", "dependency", "goal", "task_type"],
        "REASONING": ["id", "dependency", "goal", "task_type"],
        "GENERAL": ["id", "dependency", "goal", "task_type"], # For nodes without specific type
    },
    "no_type": True, # Simplifies some config lookups if task_type isn't critical
    "COMPOSITION": { # Dummy config for COMPOSITION tasks
        "atom": {},
        "planning": {},
    },
     "GENERAL": { # Dummy config for GENERAL tasks
        "atom": {},
        "planning": {},
    },
}

class TestConcurrentReadWrite(unittest.TestCase):
    def setUp(self):
        """Set up test environment: create temporary directories."""
        self.base_temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.base_temp_dir, RESULTS_DIR_NAME)
        self.records_dir = os.path.join(self.results_dir, RECORDS_DIR_NAME)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.records_dir, exist_ok=True)
        # print(f"Setup: Created temp directory {self.base_temp_dir}")

    def tearDown(self):
        """Tear down test environment: remove temporary directories."""
        shutil.rmtree(self.base_temp_dir)
        # print(f"Teardown: Removed temp directory {self.base_temp_dir}")

    def _generate_large_mock_plan(self, num_tasks=200):
        """Generates a large plan dictionary to simulate a complex LLM response."""
        plan = []
        # Create a long string for the goal. Using a portion of Lorem Ipsum.
        long_goal_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. " * 10 # Repeat to make it longer
        for i in range(num_tasks):
            plan.append({
                "id": f"task_{i}",
                "dependency": [], # Simplified dependencies
                "goal": f"Sub-task goal {i}: {long_goal_text}",
                "task_type": "write", # Or any other valid task_type
                "length": "short" # Dummy value
            })
        return plan

    def mock_monitor_read(self, nodes_file_path, stop_event, errors_list, read_counts):
        """
        Simulates a monitoring function that continuously reads the nodes.json file.
        """
        while not stop_event.is_set():
            try:
                if os.path.exists(nodes_file_path):
                    with open(nodes_file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    read_counts['success'] += 1
                else:
                    read_counts['not_found'] +=1
            except json.JSONDecodeError as e:
                errors_list.append(e)
                read_counts['failed'] += 1
            except Exception as e: # Catch other potential errors like PermissionError during rename
                errors_list.append(e)
                read_counts['failed'] += 1
            time.sleep(0.005) # Increased frequency for more chances of catching issues

    def test_concurrent_read_write_on_long_response(self):
        """
        Tests concurrent read/write operations on nodes.json with large data,
        simulating engine writes and monitor reads.
        """
        task_id = str(uuid.uuid4())
        task_record_folder = os.path.join(self.records_dir, task_id)
        os.makedirs(task_record_folder, exist_ok=True)
        
        nodes_json_file = os.path.join(task_record_folder, "nodes.json")

        # 1. Simulate long LLM response
        mock_large_plan = self._generate_large_mock_plan(num_tasks=100) # Reduced for test speed

        # 2. Initialize engine and root node
        root_node = RegularDummyNode(
            config=TEST_CONFIG,
            nid="root",
            node_graph_info={
                "outer_node": None,
                "root_node": None, # Will be self-assigned
                "parent_nodes": [],
                "layer": 0
            },
            task_info={
                "goal": "Root task goal",
                "task_type": "write", # Needs to match a type in config
                "length": "long"
            },
            node_type=NodeType.PLAN_NODE
        )
        root_node.node_graph_info["root_node"] = root_node # Self-assign root
        
        engine = GraphRunEngine(root_node=root_node, memory_format="xml", config=TEST_CONFIG)

        # 3. Setup mock monitor thread
        stop_event = threading.Event()
        read_errors = []
        read_counts = {'success': 0, 'failed': 0, 'not_found': 0}
        
        reader_thread = threading.Thread(
            target=self.mock_monitor_read,
            args=(nodes_json_file, stop_event, read_errors, read_counts)
        )
        reader_thread.start()

        # 4. Simulate engine's write operations (simplified loop)
        num_write_cycles = 20 # Number of times to simulate plan update and save
        
        for i in range(num_write_cycles):
            # Simulate engine processing that leads to plan update
            # In a real scenario, this would be more complex. Here, we directly update the plan.
            # print(f"Write cycle {i+1}/{num_write_cycles}")
            try:
                # Simulate the root node receiving a plan (e.g., from a planning agent)
                # The plan2graph method is part of AbstractNode, which RegularDummyNode inherits
                engine.root_node.raw_plan = mock_large_plan # Store the raw plan
                engine.root_node.plan2graph(mock_large_plan) # Process it into inner_graph
                engine.root_node.status = TaskStatus.PLAN_DONE # Update status to something that might trigger save

                # Simulate saving the state, which writes to nodes.json
                # The save method in GraphRunEngine writes nodes.json
                engine.save(task_record_folder) 
            except Exception as e:
                # This is to catch errors in the main thread's write operations
                print(f"Error during engine write operation: {e}") 
                # Depending on the test, we might want to fail here or record this error
            time.sleep(0.01) # Simulate some work being done by the engine

        # 5. Stop reading thread and assert
        stop_event.set()
        reader_thread.join(timeout=5) # Wait for the thread to finish

        print(f"Read counts: {read_counts}")
        print(f"Read errors: {read_errors}")

        self.assertEqual(len(read_errors), 0, 
                         f"JSON decode errors or other read errors occurred: {read_errors}")
        self.assertTrue(read_counts['success'] > 0, 
                        "The monitor thread did not successfully read the JSON file.")
        # It's possible 'not_found' > 0 if the reader starts before the first write, which is fine.

if __name__ == '__main__':
    unittest.main()
