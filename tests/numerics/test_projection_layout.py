"""连续分组、非连续数组与尾批缓冲区的必要回归。"""
import numpy as np
import pytest
from econhdfe.hdfe import projection as p
from econhdfe.compute import execution as e

@pytest.mark.parametrize('kind',['permuted_blocks','split_group'])
def test_detection(kind):
 codes=np.array([4,4,1,1,1,7,7,7,7,0],dtype=np.int32)
 if kind=='split_group':codes[-1]=4

 idx=p.build_group_index(codes,9)
 assert idx.contiguous==(kind == 'permuted_blocks')
 assert idx.nbytes==idx.order.nbytes+idx.indptr.nbytes
 assert not p.GroupIndex(idx.order,idx.indptr).contiguous


def test_tail_workspace():
 dtype=np.float64
 raw=np.arange(17*13,dtype=dtype).reshape(17,13)
 source=np.asfortranarray(raw)
 original=source.copy();target=np.zeros_like(source);ws=e.ReusableRHSWorkspace(17,5,dtype)
 ptr=ws.array.ctypes.data
 for lo,hi in [(0,5),(5,10),(10,13),(2,3),(0,5)]:
  work=ws.load(source,lo,hi)
  assert work.flags.c_contiguous and work.ctypes.data==ptr and np.shares_memory(work,ws.array)
  assert ws.array.nbytes==17*5*np.dtype(dtype).itemsize
  np.testing.assert_array_equal(work,source[:,lo:hi])
  work*=2;e.ReusableRHSWorkspace.write(target,lo,hi,work)
  np.testing.assert_array_equal(target[:,lo:hi],source[:,lo:hi]*2)
 np.testing.assert_array_equal(source,original)
