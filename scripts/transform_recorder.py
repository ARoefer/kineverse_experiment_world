#!/usr/bin/env python3
import sys

import rospy
import tf2_ros

from argparse import ArgumentParser

if __name__ == '__main__':
    rospy.init_node('transform_recorder')

    arg_parser = ArgumentParser(description='Records relative transforms from tf to a csv file')
    arg_parser.add_argument('-o', '--output', type=str, help='Output file')
    arg_parser.add_argument('-t', '--transforms', nargs='+', help='Pairs of frames (A, B) expressing "B in A"')
    arg_parser.add_argument('-r', '--rate', default=10.0, help="Rate at which the transfroms get recorded")
    args = [a for a in sys.argv[1:] if ':=' not in a]

    args = arg_parser.parse_args(args)

    if args.transforms is None or len(args.transforms) == 0:
        print('Need transforms to record')
        exit()

    if len(args.transforms) % 2 != 0:
        print(f'Need even number of transforms. Got {len(args.transforms)}')
        exit()
    
    pairs = [tuple(args.transforms[x:x+2]) for x in range(0, len(args.transforms), 2)]

    tf_buffer = tf2_ros.Buffer()
    listener  = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Duration(1 / args.rate)

    args.output = f'{args.output}.csv' if args.output[-4:].lower() != '.csv' else args.output
    out_file = open(args.output, 'w')
    out_file.write('sec, nsec, target, source, x, y, z, qx, qy, qz, qw\n')

    print('Will record:\n{}\nAt {} Hz'.format('\n  '.join(f'{s} -> {t}' for t, s in pairs), args.rate))

    while not rospy.is_shutdown():
        start = rospy.Time.now()

        for (target, source) in pairs:
            try:
                trans = tf_buffer.lookup_transform(target, source, rospy.Time(0))
                stamp = trans.header.stamp
                translation = trans.transform.translation
                rotation    = trans.transform.rotation
                out_file.write(f'{stamp.secs}, {stamp.nsecs}, '
                               f'{target}, {source}, '
                               f'{translation.x}, {translation.y}, {translation.z}'
                               f'{rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}\n')
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(f'Exception raised while looking up {source} -> {target}:\n{e}')
                continue
        
        time_remaining = rate - (rospy.Time.now() - start)
        if time_remaining > rospy.Duration(0):
            rospy.sleep(time_remaining)

    out_file.close()
    
