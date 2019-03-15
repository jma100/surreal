import bpy
import bmesh
import time
from pickle import load
from mathutils import Matrix, Vector, Quaternion, Euler
import itertools
from random import choice
import random
import numpy as np

def bmesh_copy_from_object(obj, transform=True, triangulate=True, apply_modifiers=False):
    """
    Returns a transformed, triangulated copy of the mesh
    """

    assert(obj.type == 'MESH')

    if apply_modifiers and obj.modifiers:
        me = obj.to_mesh(bpy.context.scene, True, 'PREVIEW', calc_tessface=False)
        bm = bmesh.new()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    else:
        me = obj.data
        if obj.mode == 'EDIT':
            bm_orig = bmesh.from_edit_mesh(me)
            bm = bm_orig.copy()
        else:
            bm = bmesh.new()
            bm.from_mesh(me)

    # Remove custom data layers to save memory
    for elem in (bm.faces, bm.edges, bm.verts, bm.loops):
        for layers_name in dir(elem.layers):
            if not layers_name.startswith("_"):
                layers = getattr(elem.layers, layers_name)
                for layer_name, layer in layers.items():
                    layers.remove(layer)

    if transform:
        bm.transform(obj.matrix_world)

    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)

    return bm

def bmesh_check_intersect_objects(obj, obj2):
    """
    Check if any faces intersect with the other object

    returns a boolean
    """
    assert(obj != obj2)
    # if obj2 == bpy.data.objects['Plane003']:
    #     import pdb; pdb.set_trace()

    # Triangulate
    bm = bmesh_copy_from_object(obj, transform=True, triangulate=True)
    bm2 = bmesh_copy_from_object(obj2, transform=True, triangulate=True)

    # # If bm has more edges, use bm2 instead for looping over its edges
    # # (so we cast less rays from the simpler object to the more complex object)
    # if len(bm.edges) > len(bm2.edges):
    #     bm2, bm = bm, bm2

    # Create a real mesh (lame!)
    scene = bpy.context.scene
    me_tmp = bpy.data.meshes.new(name="~temp~")
    bm2.to_mesh(me_tmp)
    bm2.free()
    obj_tmp = bpy.data.objects.new(name=me_tmp.name, object_data=me_tmp)
    scene.objects.link(obj_tmp)
    scene.update()
    ray_cast = obj_tmp.ray_cast

    intersect = False

    EPS_NORMAL = 0.000001
    EPS_CENTER = 0.01  # should always be bigger

    #for ed in me_tmp.edges:
    for ed in bm.edges:
        v1, v2 = ed.verts

        # setup the edge with an offset
        co_1 = v1.co.copy()
        co_2 = v2.co.copy()
        co_mid = (co_1 + co_2) * 0.5
        no_mid = (v1.normal + v2.normal).normalized() * EPS_NORMAL
        co_1 = co_1.lerp(co_mid, EPS_CENTER) + no_mid
        co_2 = co_2.lerp(co_mid, EPS_CENTER) + no_mid

        _, co, no, index = ray_cast(co_1, co_2)
        _, co2, no2, index2 = ray_cast(co_2, co_1)
        if index != -1 or index2 != -1:
            intersect = True
            # print('//////////////////////////////////////')
            # print(obj.name+' intersects '+obj2.name)
            break

    scene.objects.unlink(obj_tmp)
    bpy.data.objects.remove(obj_tmp)
    bpy.data.meshes.remove(me_tmp)

    scene.update()

    return intersect

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

    scene_min_x = -1
    scene_max_x = 5
    scene_min_y = -6
    scene_max_y = 1
    all_spots = [element for element in itertools.product(range(scene_min_x, scene_max_x),
                                                          range(scene_min_y, scene_max_y))]

    zrots_all = [0, 1/4*np.pi, 1/2*np.pi, 3/4*np.pi, np.pi, 5/4 * np.pi, 3/2*np.pi, 7/4*np.pi]
    random.seed(0)
    random.shuffle(all_spots)

    idx_info_lst = load(open("pkl/idx_info.pickle", 'rb'))
    idx_info_lst = [x for x in idx_info_lst if x['name'][:4] != 'h36m']

    bpy.ops.import_scene.obj(filepath='/home/jingweim/urop/bg_models/ivkwwche5p1c-001/house interior.obj')
    bpy.ops.transform.resize(value=(0.02, 0.02, 0.02))

    for i in range(11):
        bpy.ops.import_scene.obj(filepath='/home/jingweim/SURREAL/mesh/objs/'+str(i)+'.obj')
        # bpy.ops.import_scene.fbx(filepath='/home/jingweim/SURREAL/mesh/'+str(i)+'.fbx')
        name = idx_info_lst[i]['name']
        mesh = bpy.data.objects[name+'_'+name+'_mesh']
        # mesh = bpy.data.objects[name]
        # print(mesh.location)
        chosen_spot = None
        for spot in all_spots:
            mesh.location = Vector([spot[0], spot[1], 0])
            zrots = [choice(zrots_all)]
            print("//////////////////")
            print("Rotation: " + str(zrots[0]))
            print("//////////////////")
            for zrot in zrots:
                mesh.rotation_euler.z = zrot
                other_objs = [ob for ob in bpy.data.objects if ob != mesh]
                intersect = False
                for other_obj in other_objs:
                    intersect = bmesh_check_intersect_objects(mesh, other_obj) or intersect
                    # print(other_obj)
                    if intersect:
                        break
                if not intersect:
                    chosen_spot = spot
                    print("////// SPOT CHOSEN ///////")
                    print(mesh.location)

                    break
            if not intersect:
                break
            # if no intersection, skip
        if chosen_spot:
            all_spots.remove(chosen_spot)
            print('////// SPOT REMOVED ////////')
        else:
            bpy.ops.object.select_all(action='DESELECT')
            mesh.select = True
            bpy.ops.object.delete(use_global=False)
    import pdb; pdb.set_trace()
    obj = bpy.context.object
    obj2 = (ob for ob in bpy.context.selected_objects if ob != obj).__next__()
    intersect = bmesh_check_intersect_objects(obj, obj2)

    print("There are%s intersections." % ("" if intersect else " NO"))

if __name__ == '__main__':
    main()
