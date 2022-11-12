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

    ComputeBuffer mapPosCenterBuffer, trianglesBuffer;
    Vector3Int[] mapPosCenter;

    RenderTexture density, voxelIntersections;
    Camera cam;
    Light directionalLight;

    int kernelIndex;

    bool canRender;

    int width, height;
    bool renderOdds;

    #region MonoBehavior
    void Start()
    {
        kernelIndex = voxelShader.FindKernel("CSMain");

        mapPosCenterBuffer = new ComputeBuffer(1, sizeof(int) * 3);
        mapPosCenter = new Vector3Int[1];

        cam = Camera.main;
        directionalLight = FindObjectOfType<Light>();
        canRender = true;
    }

    private void Update()
    {
        Render();
    }

    private void OnDestroy()
    {
        if (mapPosCenterBuffer != null) mapPosCenterBuffer.Dispose();
        if (trianglesBuffer != null) trianglesBuffer.Dispose();
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
        InitRenderTexture();
        SetParameters();

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        canRender = false;
        voxelShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);
        StartCoroutine(GetData());

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

    IEnumerator GetData()
    {
        yield return null;
        mapPosCenterBuffer.GetData(mapPosCenter);
        /*ComputeBuffer triCountBuffer = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Raw);
        ComputeBuffer.CopyCount(trianglesBuffer, triCountBuffer, 0);
        int[] triCountArray = { 0 };
        triCountBuffer.GetData(triCountArray);
        int numTris = triCountArray[0];

        Triangle[] ts = new Triangle[numTris];
        trianglesBuffer.GetData(ts);
        trianglesBuffer.Dispose();*/

        yield return null;
        #region Job System
        Texture2D tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
        Rect rectReadPicture = new Rect(0, 0, width, height);
        RenderTexture.active = voxelIntersections;
        // Read pixels
        tex.ReadPixels(rectReadPicture, 0, 0);
        tex.Apply();
        RenderTexture.active = null; // added to avoid errors 


        NativeList<float3> vertices = new NativeList<float3>(width * height * 5 * 3, Allocator.TempJob);
        NativeList<float3> normals = new NativeList<float3>(width * height * 5 * 3, Allocator.TempJob);
        NativeList<int> indices = new NativeList<int>(width * height * 5 * 3, Allocator.TempJob);
        MeshConverterJob job = new MeshConverterJob()
        {
            vertices = vertices,
            normals = normals,
            indices = indices,
            voxelIntersections = tex.GetRawTextureData<int3>(),
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
        yield break;
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

    void SetParameters()
    {
        brushStrength = Mathf.Abs(brushStrength);
        voxelShader.SetInt("width", width);
        voxelShader.SetInt("height", height);
        voxelShader.SetTexture(kernelIndex, "Density", density);
        voxelShader.SetTexture(kernelIndex, "Destination", voxelIntersections);
        voxelShader.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix);
        voxelShader.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse);
        voxelShader.SetVector("_LightDirection", directionalLight.transform.forward);
        voxelShader.SetVector("_AmbientColor", new Vector3(ambientColor.r, ambientColor.g, ambientColor.b));
        voxelShader.SetVector("_GrassColor", new Vector3(grassColor.r, grassColor.g, grassColor.b));
        voxelShader.SetVector("_SandColor", new Vector3(sandColor.r, sandColor.g, sandColor.b));
        voxelShader.SetVector("_DirtColor", new Vector3(dirtColor.r, dirtColor.g, dirtColor.b));
        voxelShader.SetVector("_WaterColor", new Vector3(waterColor.r, waterColor.g, waterColor.b));
        voxelShader.SetFloat("_Scale", scale);
        voxelShader.SetFloat("_Impact", impact);

        int terra = 0;
        if (Input.GetMouseButton(0)) terra = 1;
        else if (Input.GetMouseButton(1)) terra = -1;
        voxelShader.SetInt("_Terraforming", terra);
        voxelShader.SetFloat("_BrushStrength", brushStrength);
        voxelShader.SetInt("_BrushSize", brushSize);

        mapPosCenterBuffer.SetData(mapPosCenter);
        voxelShader.SetBuffer(0, "mapPosCenter", mapPosCenterBuffer);

        /*trianglesBuffer = new ComputeBuffer(width * height * 6, TRIANGLE_STRUCT_SIZE, ComputeBufferType.Append);
        trianglesBuffer.SetCounterValue(0);
        voxelShader.SetBuffer(kernelIndex, "Triangles", trianglesBuffer);*/

        renderOdds = !renderOdds;
        voxelShader.SetBool("_RenderOdds", renderOdds);
    }

    void InitRenderTexture()
    {
        if (voxelIntersections == null || voxelIntersections.width != width || voxelIntersections.height != height)
        {
            if (voxelIntersections != null)
            {
                voxelIntersections.Release();
            }
            voxelIntersections = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            voxelIntersections.enableRandomWrite = true;
            voxelIntersections.Create();
        }
        Create3DTexture(ref density, 1000, "Density");
    }

    void Create3DTexture(ref RenderTexture texture, int size, string name)
    {
        //
        var format = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat;
        if (texture == null || !texture.IsCreated() || texture.width != size || texture.height != size || texture.volumeDepth != size || texture.graphicsFormat != format)
        {
            //Debug.Log ("Create tex: update noise: " + updateNoise);
            if (texture != null)
            {
                texture.Release();
            }
            const int numBitsInDepthBuffer = 0;
            texture = new RenderTexture(size, size, numBitsInDepthBuffer);
            texture.graphicsFormat = format;
            texture.volumeDepth = size;
            texture.enableRandomWrite = true;
            texture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;


            texture.Create();
        }
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Bilinear;
        texture.name = name;
    }
    #endregion
    #endregion

    #region Jobs
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
        public NativeArray<int3> voxelIntersections;
        public void Execute(int index)
        {
            
        }
    }

    #endregion
}
