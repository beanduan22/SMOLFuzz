"""Bug 53: tf.roll with shift >= 2^31 gives wrong result due to int32 overflow.
Source: TensorFlow GitHub issue #109520
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

x = tf.range(10)
shift = 2**31  # INT32_MAX+1 — overflows int32 inside TF kernel

expected = tf.roll(x, shift=shift % 10, axis=0)  # correct: shift=8
got      = tf.roll(x, shift=shift, axis=0)        # BUG: wrong

print("Expected:", expected.numpy())
print("Got:     ", got.numpy())       # BUG: differs from expected
assert not all(got.numpy() == expected.numpy()), "bug not reproduced"
print("BUG CONFIRMED: tf.roll int32 overflow with shift=2^31")
