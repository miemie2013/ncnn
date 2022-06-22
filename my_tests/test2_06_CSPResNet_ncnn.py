
'''
（在pycharm的普通终端输入）：
cd build/examples
.\test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ../../my_tests/06.param ../../my_tests/06.bin

(linux)
cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ../../my_tests/06.param ../../my_tests/06.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ../../my_tests/06_pncnn.param ../../my_tests/06_pncnn.bin


(比较输出的不同)
cd my_tests
python diff_output.py --ncnn_output ../build/examples/output.txt --torch_output 06.npz


'''



