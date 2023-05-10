import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.io import read_image

from .unet import *
import numpy as np
import cv2 as cv
import geopandas as pd
from shapely import Point, LineString
import osmnx as ox
from collections import deque

class ForestRoadsExtractor:
    
    def __init__(self, model_name, model_path, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.model_name = model_name
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()
    
    
    def preprocessing(self, image): # padding + bigbatch + transforms

        left, right, top, bottom = 0, 0, 0, 0
        if image.shape[2] < 500:
            left = (500 - image.shape[2]) // 2
            right = 500 - image.shape[2] - left
        elif image.shape[2] % 250 != 0:
            left = (250 - image.shape[2] % 250) // 2
            right = 250 - image.shape[2] % 250 - left
        if image.shape[1] < 500:
            top = (500 - image.shape[1]) // 2
            bottom = 500 - image.shape[1] - top
        elif image.shape[1] % 250 != 0:
            top = (250 - image.shape[1] % 250) // 2
            bottom = 250 - image.shape[1] % 250 - top
        padded_image = nn.functional.pad(image.float(), (left, right, top, bottom))

        in_column = padded_image.shape[1] // 250 - 1
        in_row = padded_image.shape[2] // 250 - 1
        bigbatch = torch.empty((in_column * in_row, 3, 512, 512))
        image_transform1 = transforms.Resize(512)
        image_transform2 = lambda image: (image.float() / 255.0 - 0.5) / 0.5
        for i in range(in_column):
            for j in range(in_row):
                bigbatch[i*in_row + j] = image_transform2(
                        image_transform1(
                            padded_image[:, i*250 : i*250 + 500, j*250 : j*250 + 500]
                        )
                    )

        return bigbatch, in_column, in_row, left, right, top, bottom
    
    
    @torch.no_grad()
    def processing(self, bigbatch): # minibatch + model predicting + transforms

        bigbatch_masks = torch.empty((bigbatch.shape[0], 500, 500), dtype=torch.uint8)

        self.model.eval()

        minibatch_size = 4
        image_transform = transforms.Resize(500)
        for i in range(0, bigbatch.shape[0], minibatch_size):
            minibatch = bigbatch[i : min(i + minibatch_size, bigbatch.shape[0])]
            minibatch = minibatch.to(self.device)
            if self.model_name == 'unet':
                logits = self.model(minibatch)
            else:
                logits = self.model(minibatch)['out']
            minibatch_masks = logits.argmax(dim=1)
            minibatch_masks = image_transform(minibatch_masks).cpu()
            bigbatch_masks[i : min(i + minibatch_size, bigbatch.shape[0])] = minibatch_masks

        return bigbatch_masks
    
    
    def postprocessing(self, masks, in_column, in_row, left, right, top, bottom): # parts together + unpadding
    
        mask = torch.zeros(((in_column + 1) * 250, (in_row + 1) * 250), dtype=torch.uint8)
        for i in range(in_column):
            for j in range(in_row):
                mask[i * 250 : i * 250 + 500, j * 250 : j * 250 + 500] = torch.logical_or(
                    mask[i * 250 : i * 250 + 500, j * 250 : j * 250 + 500],
                    masks[i * in_row + j]
                )
                
        mask = mask[top : mask.shape[0] - bottom, left : mask.shape[1] - right]

        return mask
    
    
    def get_raw_roads(self, image_name):
        
        image = read_image(image_name)
        bigbatch, in_column, in_row, left, right, top, bottom = self.preprocessing(image)
        masks = self.processing(bigbatch)
        mask = self.postprocessing(masks, in_column, in_row, left, right, top, bottom)
        
        return mask
    
    def dilation(self, init_mask):
        
        kernel = np.ones((2, 2), np.uint8)
        mask = cv.dilate(init_mask, kernel, iterations=1)
        
        return mask
    
    def skeletonization(self, init_mask):
        
        mask = np.copy(init_mask)
        m = np.zeros_like(mask)
        c = 1
        while c != 0:
            c = 0
            for i in range(1, mask.shape[0] - 1):
                for j in range(1, mask.shape[1] - 1):
                    if mask[i, j] == 1:
                        arr = np.array([mask[i - 1, j],
                                        mask[i - 1, j + 1], 
                                        mask[i, j + 1], 
                                        mask[i + 1, j + 1], 
                                        mask[i + 1, j], 
                                        mask[i + 1, j - 1], 
                                        mask[i, j - 1], 
                                        mask[i - 1, j - 1]])
                        a = 0
                        for k in range(len(arr)):
                            if arr[k % len(arr)] == 0 and arr[(k + 1) % len(arr)] == 1:
                                a += 1
                        if 2 <= np.sum(arr) <= 6 and \
                                a == 1 and \
                                arr[0]*arr[2]*arr[4] == 0 and \
                                arr[2]*arr[4]*arr[6] == 0:
                            m[i, j] = 1
                            c += 1
            mask -= m
            m = np.zeros_like(mask)
            if c == 0:
                break
            c = 0
            for i in range(1, mask.shape[0] - 1):
                for j in range(1, mask.shape[1] - 1):
                    if mask[i, j] == 1:
                        arr = np.array([mask[i - 1, j],
                                        mask[i - 1, j + 1], 
                                        mask[i, j + 1], 
                                        mask[i + 1, j + 1], 
                                        mask[i + 1, j], 
                                        mask[i + 1, j - 1], 
                                        mask[i, j - 1], 
                                        mask[i - 1, j - 1]])
                        a = 0
                        for k in range(len(arr)):
                            if arr[k % len(arr)] == 0 and arr[(k + 1) % len(arr)] == 1:
                                a += 1
                        if 2 <= np.sum(arr) <= 6 and \
                                a == 1 and \
                                arr[0]*arr[2]*arr[6] == 0 and \
                                arr[0]*arr[4]*arr[6] == 0:
                            m[i, j] = 1
                            c += 1
            mask -= m
            m = np.zeros_like(mask)
        
        return mask
    
    
    def get_rid_of_artifacts(self, init_mask):
        
        # (i, j) -> n : coords_inds[i, j]
        # n -> (i, j) : inds_coords[n]
        mask = np.copy(init_mask)
        x, y = np.nonzero(mask)
        n = x.shape[0]
        coords_inds = np.full(mask.shape, -1)
        coords_inds[x, y] = np.arange(n)
        inds_coords = { i : coords for i, coords in enumerate(zip(x, y))}
        
        graph = {}
        for k in range(n):
            neighbours = []
            i, j = inds_coords[k]
            if i > 0 and mask[i - 1, j] == 1:
                neighbours.append(coords_inds[i - 1, j])
            if i < mask.shape[0] - 1 and mask[i + 1, j] == 1:
                neighbours.append(coords_inds[i + 1, j])
            if j > 0 and mask[i, j - 1] == 1:
                neighbours.append(coords_inds[i, j - 1])
            if j < mask.shape[1] - 1 and mask[i, j + 1] == 1:
                neighbours.append(coords_inds[i, j + 1])
            if i > 0 and j > 0 and mask[i - 1, j - 1] == 1 and \
                    mask[i - 1, j] == 0 and mask[i, j - 1] == 0:
                neighbours.append(coords_inds[i - 1, j - 1])
            if i > 0 and j < mask.shape[1] - 1 and mask[i - 1, j + 1] == 1 and \
                    mask[i - 1, j] == 0 and mask[i, j + 1] == 0:
                neighbours.append(coords_inds[i - 1, j + 1])
            if i < mask.shape[0] - 1 and j < mask.shape[1] - 1 and mask[i + 1, j + 1] == 1 and \
                    mask[i, j + 1] == 0 and mask[i + 1, j] == 0:
                neighbours.append(coords_inds[i + 1, j + 1])
            if i < mask.shape[0] - 1 and j > 0 and mask[i + 1, j - 1] == 1 and \
                    mask[i + 1, j] == 0 and mask[i, j - 1] == 0:
                neighbours.append(coords_inds[i + 1, j - 1])
            graph[k] = neighbours
        
        visited = np.full((n, ), -1)
        n_components = 0 
        for k in range(n):
            if visited[k] == -1:
                queue = deque([k])
                while queue:
                    vertex = queue.popleft()
                    visited[vertex] = n_components
                    for neighbour in graph[vertex]:
                        if visited[neighbour] == -1:
                            queue.append(neighbour)
                n_components += 1
        
        components, counts = np.unique(visited, return_counts=True)
        to_delete = np.arange(n)[np.isin(visited, components[counts < 30])]
        for k in to_delete:
            i, j = inds_coords[k]
            mask[i, j] = 0
            coords_inds[i, j] = -1
            inds_coords.pop(k)
            visited[k] = -1
            
        return mask, inds_coords, graph
    
        
    def get_processed_roads(self, raw_roads): # dilation + skeletonization + artifacts
        
        mask = raw_roads.numpy()
        mask = self.dilation(mask)
        mask = self.skeletonization(mask)
        mask, inds_coords, graph = self.get_rid_of_artifacts(mask)
        
        return mask, inds_coords, graph
    
    def get_coords(self, k, inds_coords, coords, dx, dy):
        return coords[0][0] + inds_coords[k][0]*dx[0] + inds_coords[k][1]*dy[0], \
                coords[0][1] + inds_coords[k][0]*dx[1] + inds_coords[k][1]*dy[1]
    
    
    def get_osm_map(self, init_mask, inds_coords, graph, coords, osm_file_path):
        
        x_shape = init_mask.shape[0]
        y_shape = init_mask.shape[1]
        dx = ((coords[3][0] - coords[0][0])/x_shape, (coords[3][1] - coords[0][1])/x_shape)
        dy = ((coords[1][0] - coords[0][0])/y_shape, (coords[1][1] - coords[0][1])/y_shape)
        
        y, x, osmid = [], [], []
        for k in inds_coords.keys():
            cur_coords = self.get_coords(k, inds_coords, coords, dx, dy)
            y.append(cur_coords[0])
            x.append(cur_coords[1])
            osmid.append(k)
        nodes = pd.GeoDataFrame(columns=['y', 'x', 'osmid'])
        nodes.y = y
        nodes.x = x
        nodes.osmid = osmid
        nodes.set_index('osmid', inplace=True)

        u, v, key, geometry = [], [], [], []
        for k in inds_coords.keys():
            for neighbour in graph[k]:
                if k < neighbour:
                    u.append(k)
                    v.append(neighbour)
                    key.append(0)
                    coor1, coor2 = self.get_coords(k, inds_coords, coords, dx, dy)
                    coor3, coor4 = self.get_coords(neighbour, inds_coords, coords, dx, dy)
                    geometry.append(LineString([[coor1, coor2], [coor3, coor4]]))
        edges = pd.GeoDataFrame(columns=['u', 'v', 'key', 'geometry'], geometry='geometry', crs="EPSG:4326")
        edges.u = u
        edges.v = v
        edges.key = key
        edges.geometry = geometry
        edges.set_index(['u', 'v', 'key'], inplace=True)
        
        G = ox.utils_graph.graph_from_gdfs(nodes, edges)
        utn = ox.settings.useful_tags_node
        oxna = ox.settings.osm_xml_node_attrs
        oxnt = ox.settings.osm_xml_node_tags
        utw = ox.settings.useful_tags_way
        oxwa = ox.settings.osm_xml_way_attrs
        oxwt = ox.settings.osm_xml_way_tags
        utn = list(set(utn + oxna + oxnt))
        utw = list(set(utw + oxwa + oxwt))
        ox.settings.all_oneway = True
        ox.settings.useful_tags_node = utn
        ox.settings.useful_tags_way = utw
        ox.save_graph_xml(G, filepath=osm_file_path)
    
    def build_osm_map(self, image_name, coords, osm_file_path):
        mask = self.get_raw_roads(image_name)
        mask, inds_coords, graph = self.get_processed_roads(mask)
        self.get_osm_map(mask, inds_coords, graph, coords, osm_file_path)
