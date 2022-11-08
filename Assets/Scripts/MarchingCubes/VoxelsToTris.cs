using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public class VoxelsToTris : MonoBehaviour
{
    const int MAX_TRIS_PER_VOXEL = 6;

    #region Structs
    const int TRIANGLE_STRUCT_SIZE = sizeof(float) * 3 * 3;
    static float3 VECTOR_ERROR = new float3(1,1,1) * 0.0000001f;
    public struct Vector : IEquatable<Vector>
    {
        public float3 position;
        public float3 normal;

        public bool Equals(Vector other)
        {
            bool3 eq = other.position - this.position == VECTOR_ERROR;
            return eq.x && eq.y && eq.z;
        }
    }
    public struct Triangle : IEquatable<Triangle>
    {
#pragma warning disable 649 // disable unassigned variable warning
        public Vector c;
        public Vector b;
        public Vector a;

        public Vector this[int i]
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

    public ComputeShader voxelShader;

    ComputeBuffer mapPosCenterBuffer, trianglesBuffer;
    Vector3Int[] mapPosCenter;

    RenderTexture density;
    Camera cam;
    Light directionalLight;
    NativeArray<Triangle> tris;
    NativeList<float3> vertices;
    NativeList<int> indices;
    NativeList<float3> normals;

    int kernelIndex;

    bool canRender;

    #region MonoBehavior
    void Start()
    {
        kernelIndex = voxelShader.FindKernel("CSMain");

        mapPosCenterBuffer = new ComputeBuffer(1, sizeof(int) * 3);
        mapPosCenter = new Vector3Int[1];

        cam = Camera.main;
        directionalLight = FindObjectOfType<Light>();
        vertices = new NativeList<float3>(cam.pixelWidth * cam.pixelHeight * MAX_TRIS_PER_VOXEL * 3, Allocator.Persistent);
        normals = new NativeList<float3>(cam.pixelWidth * cam.pixelHeight * MAX_TRIS_PER_VOXEL * 3, Allocator.Persistent);
        indices = new NativeList<int>(Allocator.Persistent);

        canRender = true;
    }

    private void Update()
    {
        Render();
    }

    private void OnDestroy()
    {
        tris.Dispose();
        mapPosCenterBuffer.Dispose();
        trianglesBuffer.Dispose();
    }
    #endregion

    #region Compute Shader

    void Render()
    {
        if (!canRender) return;
        InitRenderTexture();
        SetParameters();

        int threadGroupsX = Mathf.CeilToInt(cam.pixelWidth / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(cam.pixelHeight / 8.0f);
        canRender = false;
        voxelShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);
        StartCoroutine(GetData());

    }

    IEnumerator GetData()
    {
        yield return null;
        mapPosCenterBuffer.GetData(mapPosCenter);
        ComputeBuffer triCountBuffer = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Raw);
        ComputeBuffer.CopyCount(trianglesBuffer, triCountBuffer, 0);
        int[] triCountArray = { 0 };
        triCountBuffer.GetData(triCountArray);
        int numTris = triCountArray[0];

        Triangle[] ts = new Triangle[numTris];
        trianglesBuffer.GetData(ts);
        trianglesBuffer.Dispose();

        yield return null;
        if (tris != null && tris.IsCreated) tris.Dispose();
        tris = new NativeArray<Triangle>(ts, Allocator.TempJob);
        MeshConverterJob job = new MeshConverterJob()
        {
            tris = tris,
            vertices = vertices,
            normals = normals,
            indices = indices,
        };
        JobHandle handle = job.Schedule(numTris, 64);
        handle.Complete();
        yield return null;
        tris.Dispose();

        

        //tris = new NativeArray<Triangle>(numTris, Allocator.Persistent);

        //AsyncGPUReadback.RequestIntoNativeArray(ref tris, trianglesBuffer, OnCompleteReadback);

        yield break;

        /*for (int i = 0; i < numTris; i++)
        {

            Triangle t = tris[i];
            //AddTriangle(t);
        }*/
    }

    /*void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            canRender = true;
            return;
        }
        print("Generating Triangles " + Time.time);
    }*/

    void SetParameters()
    {
        brushStrength = Mathf.Abs(brushStrength);
        voxelShader.SetInt("width", cam.pixelWidth);
        voxelShader.SetInt("height", cam.pixelWidth);
        voxelShader.SetTexture(kernelIndex, "Density", density);
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

        trianglesBuffer = new ComputeBuffer(cam.pixelWidth * cam.pixelHeight * 6, TRIANGLE_STRUCT_SIZE, ComputeBufferType.Append);
        trianglesBuffer.SetCounterValue(0);
        voxelShader.SetBuffer(kernelIndex, "Triangles", trianglesBuffer);
    }

    void InitRenderTexture()
    {
        /*if (target == null || target.width != cam.pixelWidth || target.height != cam.pixelHeight)
        {
            if (target != null)
            {
                target.Release();
            }
            target = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            target.enableRandomWrite = true;
            target.Create();
        }*/
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

    #region Jobs
    struct MeshConverterJob : IJobParallelFor
    {
        [ReadOnly]
        public NativeArray<Triangle> tris;
        [NativeDisableParallelForRestriction]
        public NativeList<float3> vertices;
        [NativeDisableParallelForRestriction]
        public NativeList<int> indices;
        [NativeDisableParallelForRestriction]
        public NativeList<float3> normals;
        public void Execute(int index)
        {
            /*Triangle triangle = tris[index];
            
            vertices.Add(triangle[0].position);
            normals.Add(triangle[0].normal);
            indices.Add(indices.Length);

            vertices.Add(triangle[1].position);
            normals.Add(triangle[1].normal);
            indices.Add(indices.Length);

            vertices.Add(triangle[2].position);
            normals.Add(triangle[2].normal);
            indices.Add(indices.Length);
            */
        }
    }

    #endregion
}
