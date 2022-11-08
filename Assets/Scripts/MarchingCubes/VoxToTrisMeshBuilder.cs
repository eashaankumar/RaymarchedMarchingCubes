using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class VoxToTrisMeshBuilder : MonoBehaviour
{
    MeshFilter meshFilter;
    
    [HideInInspector]
    public Mesh mesh;

    public static VoxToTrisMeshBuilder Instance;

    private void Awake()
    {
        Instance = this;
        meshFilter = GetComponent<MeshFilter>();
        mesh = new Mesh();
        meshFilter.mesh = mesh;
    }
}
