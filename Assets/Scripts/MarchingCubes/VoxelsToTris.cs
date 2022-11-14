using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

public class VoxelsToTris : MonoBehaviour
{
    const int MAX_TRIS_PER_VOXEL = 6;

    #region Structs
    struct Ray
    {
        public float3 origin;
        public float3 direction;
    };
    struct VoxelTris
    {
        public Triangle[] triangles;
        public int numTris;
    };
    struct RaymarchResult
    {
        public bool miss;
        public int3 mapPos;
        public VoxelTris triangles;
    };

    const int TRIANGLE_STRUCT_SIZE = sizeof(float) * 3 * 2 * 3 + sizeof(int) * 3;
    static float3 VECTOR_ERROR = new float3(1,1,1) * 0.0000001f;
    public struct Vertex : IEquatable<Vertex>
    {
        public float3 position;
        public float3 normal;
        public bool Equals(Vertex other)
        {
            bool3 eq = other.position - this.position == VECTOR_ERROR;
            return eq.x && eq.y && eq.z;
        }
    }
    public struct Triangle : IEquatable<Triangle>
    {
#pragma warning disable 649 // disable unassigned variable warning
        public Vertex c;
        public Vertex b;
        public Vertex a;
        public Vector3Int mapPos;
        public Vertex this[int i]
        {
            get
            {
                switch (i)
                {
                    case 0:
                        return a;
                    case 1:
                        return b;
                    default:
                        return c;
                }
            }
            set
            {
                switch (i)
                {
                    case 0:
                        a = value;
                        break;
                    case 1:
                        b = value;
                        break;
                    default:
                        c = value;
                        break;
                }
            }
        }

        public bool Equals(Triangle other)
        {
            return other[0].Equals(this[0]) && other[1].Equals(this[1]) && other[2].Equals(this[2]);
        }
    }
    #endregion

    [SerializeField]
    float scale;
    [SerializeField]
    float impact;
    [Header("Generator settings")]
    [SerializeField]
    Color ambientColor;
    [SerializeField]
    Color grassColor;
    [SerializeField]
    Color dirtColor;
    [SerializeField]
    Color sandColor;
    [SerializeField]
    Color waterColor;
    [SerializeField]
    float brushStrength;
    [SerializeField]
    int brushSize;

    [Header("Rendering")]
    [SerializeField, Tooltip("Voxel Detection Resolution")]
    int res;

    public ComputeShader voxelShader;

    Vector3Int[] mapPosCenter;

    Camera cam;
    Light directionalLight;

    int kernelIndex;

    bool canRender;

    int width, height;
    bool renderOdds;

    int densityGridSize = 10;
    NativeParallelHashMap<int3, float> densityMap;
    NativeArray<int3> cubeToCubeVertexArray;
    NativeArray<int2> edgeToCornersArray;
    NativeArray<float3> directionsArray;
    NativeArray<int> caseToNumPolysArray;
    NativeArray<int> edge_connect_listArray;

    #region MonoBehavior
    void Start()
    {
        kernelIndex = voxelShader.FindKernel("CSMain");
        cam = Camera.main;
        directionalLight = FindObjectOfType<Light>();
        canRender = true;
        densityMap = new NativeParallelHashMap<int3, float>(100000, Allocator.Persistent);

        cubeToCubeVertexArray = new NativeArray<int3>(cornerToCubeVertex, Allocator.Persistent);
        edgeToCornersArray = new NativeArray<int2>(edgeToCorners, Allocator.Persistent);
        directionsArray = new NativeArray<float3>(directions, Allocator.Persistent);
        caseToNumPolysArray = new NativeArray<int>(caseToNumPolys, Allocator.Persistent);
        edge_connect_listArray = new NativeArray<int>(edge_connect_list, Allocator.Persistent);
    }

    private void Update()
    {
        Render();
    }

    private void OnDestroy()
    {
    }
    #endregion

    #region Compute Shader
    void Init()
    {
        res = Mathf.Max(res, 1);
        width = cam.pixelWidth / res;
        height = cam.pixelHeight / res;
    }
    void Render()
    {
        if (!canRender) return;
        Init();
        //InitRenderTexture();
        //SetParameters();

        /*int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        canRender = false;
        voxelShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);*/
        //StartCoroutine(GetData());
        GetData();

    }

    // https://answers.unity.com/questions/1819160/cast-vector3-to-float3.html
    public unsafe static void MemCpy<SRC, DST>(NativeArray<SRC> src, DST[] dst)
    where SRC : struct
    where DST : struct
    {
        int srcSize = src.Length * UnsafeUtility.SizeOf<SRC>();
        int dstSize = dst.Length * UnsafeUtility.SizeOf<DST>();
        Assert.AreEqual(srcSize, dstSize, $"{nameof(srcSize)}:{srcSize} and {nameof(dstSize)}:{dstSize} must be equal.");
        void* srcPtr = NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(src);
        void* dstPtr = UnsafeUtility.PinGCArrayAndGetDataAddress(dst, out ulong handle);
        UnsafeUtility.MemCpy(destination: dstPtr, source: srcPtr, size: srcSize);
        UnsafeUtility.ReleaseGCObject(handle);
    }

    unsafe void GetData()
    {
        #region Job System
        // Grab voxel intersections
        NativeList<float3> vertices = new NativeList<float3>(width * height * 5 * 3, Allocator.TempJob);
        NativeList<float3> normals = new NativeList<float3>(width * height * 5 * 3, Allocator.TempJob);
        NativeList<int> indices = new NativeList<int>(width * height * 5 * 3, Allocator.TempJob);

        MeshConverterJob job = new MeshConverterJob()
        {
            vertices = vertices,
            normals = normals,
            indices = indices,
            densityMap = densityMap,
            width = width,
            height = height,
            maxStepCount = 500,
            _CameraToWorld = cam.cameraToWorldMatrix,
            _CameraInverseProjection = cam.projectionMatrix.inverse,
            cornerToCubeVertex = cubeToCubeVertexArray,
            edgeToCorners = edgeToCornersArray,
            directions = directionsArray,
            caseToNumPolys = caseToNumPolysArray,
            edge_connect_list = edge_connect_listArray
        };
        JobHandle handle = job.Schedule(width * height, 64);
        handle.Complete();

        //VoxToTrisMeshBuilder.Instance.mesh.Clear();

        //VoxToTrisMeshBuilder.Instance.mesh.vertices = new Vector3[vertices.Length];
        //MemCpy<float3, Vector3>(vertices, VoxToTrisMeshBuilder.Instance.mesh.vertices);
        //VoxToTrisMeshBuilder.Instance.mesh.SetVertices(vertices.ToArray());
        //VoxToTrisMeshBuilder.Instance.mesh.SetNormals(normals.ToArray());
        //VoxToTrisMeshBuilder.Instance.mesh.SetTriangles(indices.ToArray(), 0);

        vertices.Dispose();
        normals.Dispose();
        indices.Dispose();
        #endregion
        #region Async readback fails
        //tris = new NativeArray<Triangle>(numTris, Allocator.Persistent);

        //AsyncGPUReadback.RequestIntoNativeArray(ref tris, trianglesBuffer, OnCompleteReadback);
        #endregion

        #region Manual Reconstruction of Triangles
        /*List<Vector3> vertices = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();
        List<int> indices = new List<int>();
        HashSet<Vector3Int> seenVoxels = new HashSet<Vector3Int>();
        for (int i = 0; i < numTris; i++)
        {

            Triangle t = ts[i];
            if (!seenVoxels.Contains(t.mapPos))
            {
                seenVoxels.Add(t.mapPos);
                for (int j = 2; j >= 0; j--)
                {
                    vertices.Add(t[j].position);
                    normals.Add(t[j].normal);
                    indices.Add(indices.Count);
                }
            }
        }

        VoxToTrisMeshBuilder.Instance.mesh.Clear();
        VoxToTrisMeshBuilder.Instance.mesh.SetVertices(vertices.ToArray());
        VoxToTrisMeshBuilder.Instance.mesh.SetNormals(normals.ToArray());
        VoxToTrisMeshBuilder.Instance.mesh.SetTriangles(indices.ToArray(), 0);
        */
        #endregion

        canRender = true;
    }

    #region Helpers
    void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            canRender = true;
            return;
        }
        print("Generating Triangles " + Time.time);
        //tris.Dispose();
        //trianglesBuffer.Dispose();

        canRender = true;
    }
    #endregion
    #endregion

    #region Jobs
    static float3[] directions = {
            new float3(1, 1, 0),
            new float3(-1, 1, 0),
            new float3(1,-1, 0),
            new float3(-1,-1, 0),
            new float3(1, 0, 1),
            new float3(-1, 0, 1),
            new float3(1, 0,-1),
            new float3(-1, 0,-1),
            new float3(0, 1, 1),
            new float3(0,-1, 1),
            new float3(0, 1,-1),
            new float3(0,-1,-1),
            new float3(1, 1, 0),
            new float3(-1, 1, 0),
            new float3(0,-1, 1),
            new float3(0,-1,-1)
        };

    static int[] caseToNumPolys = {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4,
            2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1,
            3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0
        };

    // 256 rows, 5 vec4s (ignore 4th component) representing edges
    static int[] edge_connect_list = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1,
            3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1,
            3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1,
            3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1,
            9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1,
            1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1,
            9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1,
            2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1,
            8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1,
            9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1,
            4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1,
            3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1,
            1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1,
            4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1,
            4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1,
            9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1,
            1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1,
            5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1,
            2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1,
            9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1,
            0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1,
            2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1,
            10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1,
            4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1,
            5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1,
            5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1,
            9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1,
            0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1,
            1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1,
            10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1,
            8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1,
            2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1,
            7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1,
            9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1,
            2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1,
            11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1,
            9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1,
            5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0,
            11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0,
            11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1,
            1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1,
            9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1,
            5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1,
            2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1,
            0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1,
            5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1,
            6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1,
            0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1,
            3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1,
            6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1,
            5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1,
            1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1,
            10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1,
            6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1,
            1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1,
            8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1,
            7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,
            3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1,
            5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1,
            0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1,
            9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6,
            8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1,
            5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11,
            0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7,
            6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1,
            10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1,
            10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1,
            8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1,
            1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1,
            3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1,
            0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1,
            10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1,
            0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1,
            3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1,
            6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1,
            9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1,
            8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1,
            3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1,
            6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1,
            0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1,
            10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1,
            10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1,
            1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1,
            2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,
            7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1,
            7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1,
            2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7,
            1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11,
            11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1,
            8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6,
            0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1,
            7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1,
            10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1,
            2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1,
            6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1,
            7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1,
            2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1,
            1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1,
            10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1,
            10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1,
            0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1,
            7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1,
            6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1,
            8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1,
            9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1,
            6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1,
            1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1,
            4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1,
            10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3,
            8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1,
            0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1,
            1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1,
            8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1,
            10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1,
            4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3,
            10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1,
            5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1,
            11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1,
            9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1,
            6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1,
            7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1,
            3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6,
            7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1,
            9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1,
            3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1,
            6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,
            9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1,
            1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,
            4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10,
            7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1,
            6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1,
            3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1,
            0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1,
            6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1,
            1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1,
            0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10,
            11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5,
            6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1,
            5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1,
            9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1,
            1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,
            1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6,
            10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1,
            0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1,
            5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1,
            10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1,
            11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1,
            0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1,
            9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1,
            7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2,
            2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1,
            8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1,
            9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1,
            9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2,
            1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1,
            9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1,
            9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1,
            5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1,
            0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1,
            10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4,
            2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1,
            0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11,
            0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5,
            9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1,
            5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1,
            3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9,
            5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1,
            8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1,
            0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1,
            9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1,
            0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1,
            1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1,
            3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4,
            4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1,
            9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3,
            11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1,
            11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1,
            2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1,
            9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7,
            3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10,
            1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1,
            4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1,
            4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1,
            0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1,
            3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1,
            3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1,
            0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1,
            9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1,
            1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            };

    static int3[] cornerToCubeVertex = {
            new int3(0, 0, 0),
            new int3(0, 1, 0),
            new int3(1, 1, 0),
            new int3(1, 0, 0),
            new int3(0, 0, 1),
            new int3(0, 1, 1),
            new int3(1, 1, 1),
            new int3(1, 0, 1)
        };
    
    static int2[] edgeToCorners = {
            new int2(0, 1),
            new int2(1, 2),
            new int2(3, 2),
            new int2(0, 3),
            new int2(4, 5),
            new int2(5, 6),
            new int2(7, 6),
            new int2(4, 7),
            new int2(0, 4),
            new int2(1, 5),
            new int2(2, 6),
            new int2(3, 7)
        };
    [BurstCompile]
    struct MeshConverterJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeList<float3> vertices;
        [NativeDisableParallelForRestriction]
        public NativeList<float3> normals;
        [NativeDisableParallelForRestriction]
        public NativeList<int> indices;
        [ReadOnly]
        public NativeParallelHashMap<int3,float> densityMap;
        [ReadOnly]
        public static float planetRadius;
        [ReadOnly]
        public static float planetCenter;
        [ReadOnly]
        public static float noisePosition;
        [ReadOnly]
        public int width;
        [ReadOnly]
        public int height;
        [ReadOnly]
        public int maxStepCount;
        [ReadOnly]
        public float4x4 _CameraToWorld;
        [ReadOnly]
        public float4x4 _CameraInverseProjection;

        int2 getId(int index)
        {
            return new int2(index % width, index / width);
        }

        public void Execute(int index)
        {
            int2 id = getId(index);
            float2 uv = id.xy / new float2(width, height);
            // Raymarching:
            Ray ray = CreateCameraRay(uv * 2 - 1);

            float4 result = new float4( ray.direction * 0.5f + 0.5f, 0);
            RaymarchResult raymarchResult = raymarchDDA(ray.origin, ray.direction, maxStepCount);
            if (!raymarchResult.miss)
            {
                CreateTrisForCube(raymarchResult.mapPos);
            }
        }


        Ray CreateRay(float3 origin, float3 direction)
        {
            Ray ray;
            ray.origin = origin;
            ray.direction = direction;
            return ray;
        }

        Ray CreateCameraRay(float2 uv)
        {
            float3 origin = math.mul(_CameraToWorld, new float4(0, 0, 0, 1)).xyz;
            float3 direction = math.mul(_CameraInverseProjection, new float4(uv, 0, 1)).xyz;
            direction = math.mul(_CameraToWorld, new float4(direction, 0)).xyz;
            direction = math.normalize(direction);
            return CreateRay(origin, direction);
        }

        RaymarchResult raymarchDDA(float3 o, float3 dir, int maxStepCount)
        {
            // https://www.shadertoy.com/view/4dX3zl
            float3 p = o;
            // which box of the map we're in
            int3 mapPos = new int3(math.floor(p));
            // length of ray from one xyz-side to another xyz-sideDist
            float3 deltaDist = math.abs(new float3(1, 1, 1) * math.length(dir) / dir);
            int3 rayStep = new int3(math.sign(dir));
            // length of ray from current position to next xyz-side
            float3 sideDist = (math.sign(dir) * (math.float3(mapPos.x, mapPos.y, mapPos.z) - o) + (math.sign(dir) * 0.5f) + 0.5f) * deltaDist;
            bool3 mask;
            bool miss = false;
            float pathLength = 0;
            RaymarchResult res = new RaymarchResult();
            for (int i = 0; i < maxStepCount; i++)
            {
                //hits = hitsSurface(mapPos, o + dir * pathLength, dir, o + getVoxelExitOffset(sideDist, deltaDist));
                VoxelTris triangles = CreateTrisForCube(mapPos);
                if (triangles.numTris > 0)
                {
                    res.triangles = triangles;
                    break;
                }
                if (sideDist.x < sideDist.y)
                {
                    if (sideDist.x < sideDist.z)
                    {
                        pathLength = sideDist.x;
                        sideDist.x += deltaDist.x;
                        mapPos.x += rayStep.x;
                        //mask = bool3(true, false, false);
                    }
                    else
                    {
                        pathLength = sideDist.z;
                        sideDist.z += deltaDist.z;
                        mapPos.z += rayStep.z;
                        //mask = bool3(false, false, true);
                    }
                }
                else
                {
                    if (sideDist.y < sideDist.z)
                    {
                        pathLength = sideDist.y;
                        sideDist.y += deltaDist.y;
                        mapPos.y += rayStep.y;
                        //mask = bool3(false, true, false);
                    }
                    else
                    {
                        pathLength = sideDist.z;
                        sideDist.z += deltaDist.z;
                        mapPos.z += rayStep.z;
                        //mask = bool3(false, false, true);
                    }
                }
                if (i == maxStepCount - 1)
                {
                    miss = true;
                }
            }
            res.miss = miss;
            res.mapPos = mapPos;
            return res;
        }

        VoxelTris CreateTrisForCube(int3 id)
        {
            int caseByte = cellCase(id);
            int numTris = caseToNumPolys[caseByte];
            VoxelTris voxelTris = new VoxelTris();
            voxelTris.numTris = numTris;
            voxelTris.triangles = new Triangle[numTris];
            for (int t = 0; t < numTris; t++)
            {
                int[] edgesOfTri = fromVec2ECL(caseByte, t);

                Vertex triVertexOnEdge1 = VertexFromInterpolatedNoise(edgesOfTri[0], id);
                Vertex triVertexOnEdge2 = VertexFromInterpolatedNoise(edgesOfTri[1], id);
                Vertex triVertexOnEdge3 = VertexFromInterpolatedNoise(edgesOfTri[2], id);

                // add tris
                Triangle triangle = new Triangle();
                triangle.a = triVertexOnEdge1;
                triangle.b = triVertexOnEdge2;
                triangle.c = triVertexOnEdge3;
                voxelTris.triangles[t] = triangle;
            }
            return voxelTris;
        }

        float sampleDensity(int3 p)
        {
            if (densityMap.ContainsKey(p)) return densityMap[p];
            return 0;
        }

        float density(int3 p)
        {
            /*float3 p = float3(c)+float3(1, 1, 1) * 0.5;
            const float3 off = float3(2134, 213, 24);
            float d = distance(p, planetCtr);
            float3 pp = p + noisePosition;
            float noise = lerpF(-1.0, 1.0, d / (planetRadius * 2));
            return noise + (ridgedNoise2(normalize(pp * surfaceNoiseScale), 3, 2.0, 2.0) * 2 - 1) * surfaceNoiseImpact;*/

            float3 worldPos = p;
            float d = math.distancesq(worldPos, planetCenter);
            float noise = math.lerp(-1, 1, math.saturate(d / (planetRadius * 2)));
            float3 pp = worldPos + noisePosition;
            float den = sampleDensity(p);
            //return noise + ridgedNoise2(normalize(pp * surfaceNoiseScale), 3, 2.0, 2.0) * surfaceNoiseImpact;
            //return noise + ridgedNoise(normalize(pp * _Scale)) * _Impact;
            return noise + den;
        }

        int getBit(int3 v)
        {
            return density(v) < 0 ? 0 : 1;
        }

        int cellCase(int3 v0)
        {
            int v0x = v0.x, v0y = v0.y, v0z = v0.z;
            int3 v1 = new int3(v0x, v0y + 1, v0z);
            int3 v2 = new int3(v0x + 1, v0y + 1, v0z);
            int3 v3 = new int3(v0x + 1, v0y, v0z);
            int3 v4 = new int3(v0x, v0y, v0z + 1);
            int3 v5 = new int3(v0x, v0y + 1, v0z + 1);
            int3 v6 = new int3(v0x + 1, v0y + 1, v0z + 1);
            int3 v7 = new int3(v0x + 1, v0y, v0z + 1);

            int caseByte = getBit(v7) << 7 | getBit(v6) << 6 |
                getBit(v5) << 5 | getBit(v4) << 4 |
                getBit(v3) << 3 | getBit(v2) << 2 |
                getBit(v1) << 1 | getBit(v0);
            return caseByte;
        }

        Vertex VertexFromInterpolatedNoise(int edgeId, int3 cubePos)
        {
            Vertex v;
            int2 edgeCorners = edgeToCorners[edgeId];
            int3 cornerA = CubeCornersToChunkNoiseGridPoints(edgeCorners.x, cubePos);
            int3 cornerB = CubeCornersToChunkNoiseGridPoints(edgeCorners.y, cubePos);
            float time = InterpolateTriangleVertexOnCubeEdge(cornerA, cornerB);
            float3 cornerAWorld = NoiseGridToWorldPos(cornerA);
            float3 cornerBWorld = NoiseGridToWorldPos(cornerB);
            v.position = math.lerp(cornerAWorld, cornerBWorld, time);

            float3 normalA = calculateNormal(cornerA);
            float3 normalB = calculateNormal(cornerB);
            v.normal = math.normalize(math.lerp(normalA, normalB, time));
            return v;
        }

        [ReadOnly]
        public NativeArray<int3> cornerToCubeVertex;
        [ReadOnly]
        public NativeArray<int2> edgeToCorners;
        [ReadOnly]
        public NativeArray<float3> directions;
        [ReadOnly]
        public NativeArray<int> caseToNumPolys;
        [ReadOnly]
        public NativeArray<int> edge_connect_list;

        int3 CubeCornersToChunkNoiseGridPoints(int corner, int3 cubePos)
        {
            return cornerToCubeVertex[corner] + cubePos;
        }

        float invLerp(float from, float to, float value)
        {
            return (value - from) / (to - from);
        }

        float InterpolateTriangleVertexOnCubeEdge(int3 cornerA, int3 cornerB)
        {
            return invLerp(density(cornerA), density(cornerB), 0);
        }

        float3 NoiseGridToWorldPos(int3 p)
        {
            return new float3(p.x, p.y, p.z);
        }

        float3 calculateNormal(int3 coord)
        {
            int2 normalOffset = new int2(1, 0);
            float dx = density(coord + normalOffset.xyy) - density(coord - normalOffset.xyy);
            float dy = density(coord + normalOffset.yxy) - density(coord - normalOffset.yxy);
            float dz = density(coord + normalOffset.yyx) - density(coord - normalOffset.yyx);
            return math.normalize(new float3(dx, dy, dz));
            //return normalize(coord);
        }

        /// <summary>
        /// Returns edge triplet given case if of voxel and poligon index desired from edge_connect_list
        /// </summary>
        /// <param name="caseId">[0,256)</param>
        /// <param name="polygonI">[0, 4)</param>
        /// <returns></returns>
        int[] fromVec2ECL(int caseId, int polygonI)
        {
            int[] edges = new int[3];
            int row = caseId;
            int col = polygonI * 3; // first element of 3d vector
            int index = col + row * 15;
            edges[0] = edge_connect_list[index];
            edges[1] = edge_connect_list[index + 1];
            edges[2] = edge_connect_list[index + 2];
            return edges;
        }

        #endregion
    }
}
