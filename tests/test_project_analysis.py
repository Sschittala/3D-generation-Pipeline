"""Independent topology controls: tetrahedron, open triangle, disconnected shells."""
import unittest
import numpy as np
from scripts.build_project_analysis import topology


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.f = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])

    def test_closed_tetrahedron(self):
        result,_ = topology(self.v,self.f)
        self.assertTrue(result['watertight_edges'])
        self.assertEqual(result['connected_components'],1)
        self.assertEqual(result['boundary_edges'],0)
        self.assertEqual(result['degenerate_faces'],0)

    def test_open_triangle(self):
        result,_ = topology(self.v,self.f[:1])
        self.assertFalse(result['watertight_edges'])
        self.assertEqual(result['boundary_edges'],3)
        self.assertEqual(result['connected_components'],1)  # unused vertex excluded

    def test_two_disconnected_shells(self):
        result,_ = topology(np.vstack([self.v,self.v+3]),np.vstack([self.f,self.f+4]))
        self.assertEqual(result['connected_components'],2)
        self.assertEqual(result['largest_component_vertex_fraction'],.5)

    def test_nonmanifold_edge(self):
        result,_ = topology(self.v,np.array([[0,1,2],[0,1,3],[0,1,2]]))
        self.assertGreater(result['nonmanifold_edges'],0)


if __name__ == '__main__':
    unittest.main()
