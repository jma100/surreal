import time
import bpy
import bmesh
from pickle import load
import numpy as np
import random
from random import choice
from os.path import join
from mathutils import Matrix, Vector, Quaternion, Euler
import math

# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}


start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))


# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses': smpl_data[seq],
                                                   'trans': smpl_data[seq.replace('pose_', 'trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return (cmu_parms, fshapes, name)

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)


# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, vertices, vertices_face, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    tmp = []
    tmp_face = []
    tmp_face2 = []
    if frame is not None:
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone_name = part_match['bone_%02d' % ibone]
        bone = arm_ob.pose.bones[obname+'_'+bone_name]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

            if frame > 0:
                # if bone_name in ['Head', 'L_Hand', 'R_Hand', 'L_Foot', 'R_Foot']:
                #     vertices.append(arm_ob.pose.bones[obname+'_'+bone_name].location)
                # vertices.append(arm_ob.pose.bones[obname+'_'+bone_name].location)
                # if bone_name in ['Head']:
                v = arm_ob.matrix_world.copy() *arm_ob.pose.bones[obname+'_'+bone_name].head.copy()
                tmp.append(v)
                if bone_name in ['Head', 'L_Foot', 'R_Foot']:
                    tmp_face.append(v)
                if bone_name in ['Head', 'L_Hand', 'R_Hand']:
                    tmp_face2.append(v)
                # vertices.append(v)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)
    if tmp != []:
        vertices.append(tmp)
    if tmp_face != []:
        vertices_face.append(tmp_face)
        vertices_face.append(tmp_face2)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, [], [])

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)


def main():
    # time logging
    global start_time
    start_time = time.time()

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)
    bpy.data.objects['Camera'].select = True
    bpy.ops.object.delete(use_global=False)
    bpy.data.objects['Lamp'].select = True
    bpy.ops.object.delete(use_global=False)

    # load config and data
    ishape = 0
    stride = 50
    stepsize = 4  # subsampling MoCap sequence by selecting every 4th frame
    clipsize = 100  # nFrames in each clip, where the random parameters are fixed
    scene = bpy.data.scenes['Scene']

    smpl_data_folder = '/home/jingweim/urop/surreal/datageneration/smpl_data/SURREAL/smpl_data'
    smpl_data_filename = 'smpl_data.npz'
    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

    idx_info_lst = load(open("pkl/idx_info.pickle", 'rb'))
    idx_info_lst = [x for x in idx_info_lst if x['name'][:4] != 'h36m']
    total = len(idx_info_lst)

    genders = {0: 'female', 1: 'male'}

    for idx in range(total):
        ### initialize for idx ###
        (runpass, idx) = divmod(idx, total)
        idx_info = idx_info_lst[idx]

        # initialize RNG with seeds from sequence id
        import hashlib
        s = "synth_data:%d:%d:%d" % (idx, runpass, ishape)
        seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
        random.seed(seed_number)
        np.random.seed(seed_number)

        # pick random gender
        gender = choice(genders)


        ### import synthetic humans ###
        bpy.ops.import_scene.fbx(
            filepath=join(smpl_data_folder, 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
            axis_forward='Y', axis_up='Z', global_scale=100)
        obname = '%s_avg' % gender[0]
        ob = bpy.data.objects[obname]
        ob.data.use_auto_smooth = False  # autosmooth creates artifacts

        # clear existing animation data
        ob.data.shape_keys.animation_data_clear()
        arm_ob = bpy.data.objects['Armature']
        arm_ob.animation_data_clear()

        cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)

        log_message("Loaded body data for %s" % name)

        nb_fshapes = len(fshapes)
        if idx_info['use_split'] == 'train':
            fshapes = fshapes[:int(nb_fshapes * 0.8)]
        elif idx_info['use_split'] == 'test':
            fshapes = fshapes[int(nb_fshapes * 0.8):]

        # pick random real body shape
        shape = choice(fshapes)
        scene.objects.active = arm_ob
        orig_trans = np.asarray(arm_ob.pose.bones[obname + '_Pelvis'].location).copy()

        data = cmu_parms[name]

        fbegin = ishape * stepsize * stride
        fend = min(ishape * stepsize * stride + stepsize * clipsize, len(data['poses']))

        reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                              smpl_data['regression_verts'], smpl_data['joint_regressor'])
        random_zrot = 0
        first_frame_trans = [0, 0, 0]
        verts = []
        verts_face = []
        arm_ob.rotation_euler.x -= math.pi / 2

        for seq_frame, (pose, trans) in enumerate(
                zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
            scene.frame_set(seq_frame)
            if seq_frame == 0:
                first_frame_trans = trans.copy()
                first_frame_trans[2] = 0
                first_frame_trans[1] += 1
            # apply the translation, pose and shape to the character
            apply_trans_pose_shape(Vector(trans - first_frame_trans), pose, shape, ob, arm_ob,
                                   obname, scene, verts, verts_face, seq_frame)

            arm_ob.pose.bones[obname + '_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
            arm_ob.pose.bones[obname + '_root'].keyframe_insert('rotation_quaternion', frame=seq_frame)

            scene.update()

        bpy.ops.object.select_all(action='DESELECT')

        ob.select = True
        bpy.ops.object.delete(use_global=False)
        arm_ob.select = True
        bpy.ops.object.delete(use_global=False)

        mesh = bpy.data.meshes.new(name+"_mesh")
        obj = bpy.data.objects.new(name, mesh)
        bm = bmesh.new()

        scene = bpy.context.scene
        scene.objects.link(obj)  # put the object into the scene (link)
        scene.objects.active = obj  # set as the active object in the scene
        obj.select = True  # select object

        # Uncomment for obj
        num_bones = range(len(verts[0]))
        count = 0
        for i, v in enumerate(verts[1:]):
            for bone in num_bones:
                v1 = bm.verts.new(v[bone])
                v2 = bm.verts.new(verts[i][bone])
                bm.edges.new((v1, v2))  # add a new edge
                count += 1

        for v in verts_face:
            v1 = bm.verts.new(v[0])
            v2 = bm.verts.new(v[1])
            v3 = bm.verts.new(v[2])

            bm.faces.new((v1, v2, v3))

        # # Uncomment for fbx
        # for v in verts:
        #     bm.verts.new(v)

        bm.to_mesh(mesh)
        bm.free()
        # import pdb; pdb.set_trace()
        import os


        target_file = os.path.join('/home/jingweim/SURREAL/mesh/objs/', str(idx)+'.obj')
        bpy.ops.export_scene.obj(filepath=target_file, use_selection=True, use_materials=False)

        # target_file = os.path.join('/home/jingweim/SURREAL/mesh', str(idx)+'.fbx')
        # bpy.ops.export_scene.fbx(filepath=target_file, use_selection=True)

        obj.select = True
        bpy.ops.object.delete(use_global=False)

        if idx > 9:
            import pdb; pdb.set_trace()
    # loop through all idxs
        # create mesh, obj

        # loop through 100 frames added vertices for each of
        # {Head, L_foot, R_foot, L_wrist, R_wrist, Pelvis} into mesh
        # should be moved to origin

        # export mesh to fbx


if __name__ == '__main__':
    main()





# verts = [(1, 1, 1), (0, 0, 0)]  # 2 verts made with XYZ coords
# mesh = bpy.data.meshes.new("mesh")  # add a new mesh
# obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh
#
# scene = bpy.context.scene
# scene.objects.link(obj)  # put the object into the scene (link)
# scene.objects.active = obj  # set as the active object in the scene
# obj.select = True  # select object
#
# mesh = bpy.context.object.data
# bm = bmesh.new()
#
# for v in verts:
#     bm.verts.new(v)  # add a new vert
#
# # make the bmesh the object's mesh
# bm.to_mesh(mesh)
# bm.free()
#
#
# import os
#
# blend_file_path = bpy.data.filepath
# directory = os.path.dirname(blend_file_path)
# target_file = os.path.join(directory, 'myfile.obj')
#
# bpy.ops.export_scene.fbx(filepath=target_file, use_selection=True)
#
# # bpy.ops.export_scene.obj(filepath="", check_existing=True, axis_forward='-Z', axis_up='Y',
# #                          filter_glob="*.obj;*.mtl", use_selection=False, use_animation=False,
# #                          use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False,
# #                          use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True,
# #                          use_materials=True, use_triangles=False, use_nurbs=False,
# #                          use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
# #                          group_by_material=False, keep_vertex_order=False, global_scale=1,
# #                          path_mode='AUTO')