В папке forest_roads_extractor находится код класса ForestRoadsExtractor.

В [ноутбуке](https://github.com/AnyaAkhmatova/ForestRoadsExtractor/blob/main/forest_roads_extractor/forest_roads_extractor_example.ipynb) содержится пример того, как можно пользоваться классом.

Изображения report_image.jpg, report_image1.jpg и report_image2.jpg можно использовать для тестирования класса. Координаты углов для этих изображений: 
- для report_image.jpg - ((53.24962646775486, -117.000305),
(53.248851201797216, -116.88792487704917),
(53.18142768480264, -116.88792385245901),
(53.182202950760285, -117.00030397540984));
- для report_image1.jpg - ((46.94616825397537, 22.314035280374384),
(46.944417443471046, 22.412501785544453),
(46.8769376801115, 22.410893096224296),
(46.87868849061583, 22.312426591054226));
- для report_image2.jpg - ((57.74229815282701, 56.999664053339856),
(57.74138183324759, 57.125606764648786),
(57.67400835746346, 57.12560736950871),
(57.67492467704288, 56.99966465819978)).

Изображения report_image_osm.jpg, report_image1_osm.jpg и report_image2_osm.jpg содержат osm-карты для этих участков.

Для работы класса необходимы следующие библиотеки: pytorch, torchvision, numpy, opencv, osmnx. К нему прилагается код модели UNet, который был написан самостоятельно, в torchvision такой модели нет.

Методы класса ForestRoadsExtractor:
- ``` def __init__(self, model_name, model_path, device=torch.device(’cuda:0’ if torch.cuda.is_available() else ’cpu’))``` – метод, инициализирующий класс, требует указания названия модели model_name (‘fcn_resnet50’, ‘unet’, ‘deeplabv3_resnet50’), пути к файлу с моделью model_path, дополнительно можно указать device, на котором будут производится вычисления;
- ``` def get_raw_roads(self, image_name)``` – метод, который строит маску для изображения, требует указания пути до файла image_name (изображение должно быть можно считать с помощью функции torchvision.io.read_image, изображение должно показывать лесную местность в весенний/летний/осенний период, без облаков, с разрешением 1 пк ∼ 10 м), возвращает двумерный массив с классами для пикселей (0 - фон и 1 - дорога); метод вызывает последовательно другие методы: preprocessing, processing и postprocessing; preprocessing делает padding изображения, чтобы его высота и ширина были больше 500 и делились на 250, проходит по нему с окном 500x500 с шагом 250, для кусочков делает Resize(512) и нормировку, сохраняет их в тензор bigbatch; processing пропускает bigbatch через модель, делает argmax для логитов и получает классы для пикселей, делает обратный Resize(500) и сохраняет кусочки в массив bigbatch_masks; postprocessing склеивает части вместе, если для пикселя модель хотя бы в одном из кусочков предсказала дорогу, то пикселю будет присвоен класс 1 в выходной маске, если во всех кусочках, покрывающих пиксель, модель предсказала для него фон, то пикселю будет присвоен класс 0, также метод делает unpadding;
- ``` def get_processed_roads(self, raw_roads)``` – метод, обрабатывающий предсказанную маску, призванный улучшить качество выхода модели, требует на вход маску raw_roads, полученную методом get_raw_roads, возвращает обработанную маску и вспомогательные для построения векторной карты в формате osm словарь inds_coords
(ставящий в соответствие номерам вершин в графе, которыми являются пиксели с классом 1, их координаты в маске) и граф graph (списки смежности); метод вызывает
последовательно другие методы: dilation, skeletonization и get_rid_of_artifacts; метод dilation расширяет дороги морфологическими методами (ядром 2x2) для возможного
восстановления связности сети дорог; метод skeletonization сужает дороги так, чтобы они были шириной 1 пиксель, метод описан в [статье](https://dl.acm.org/doi/10.1145/357994.358023), он итеративно убирает сначала правые нижние границы, затем левые верхние границы, пока находятся пиксели,
подпадающие под условия удаления из изображения; метод get_rid_of_artifacts строит по получившейся маске граф, он нумерует оставшиеся пиксели с классом 1, создает
словарь inds_coords, считает эти пиксели вершинами и строит ребра между смежными пикселями дороги, обходит граф в ширину, каждому пикселю дороги присваивает
компоненту связности, затем компоненты связности мощности меньше 30 (длина дорог меньше 300 м) стирает с маски, как артефакты;
- ``` def get_osm_map(self, init_mask, inds_coords, graph, coords, osm_file_path)``` – метод, строящий векторную карту в формате osm, требует на вход маску init_mask,
словарь inds_coords, граф graph, полученные методом get_processed_roads, а также координаты углов изображения coords в порядке: верхний левый, верхний правый, нижний правый, нижний левый, в формате (широта, долгота), и название для osm-файла osm_file_path; метод линейно вычисляет географические координаты всех пикселей
изображения по координатам внутри массива и географическим координатам верхнего левого, верхнего правого и нижнего левого краев снимка; метод с помощью geopandas
строит GeoDataFrame nodes для вершин и GeoDataFrame edges для ребер, с помощью функции osmnx.utils_graph.graph_from_gdfs строит по ним networkx.MultiDiGraph G,
сохраняет его, как osm-файл, с помощью функции osmnx.save_graph_xml;
- ``` def build_osm_map(self, image_name, coords, osm_file_path)``` – метод, который по спутниковому снимку и координатам его углов строит векторную карту дорог и сохраняет ее в osm-файл, требует указать путь к снимку image_name, координаты углов изображения coords, название для osm-файла osm_file_path; метод последовательно вызывает другие методы: get_raw_roads, get_processed_roads и get_osm_map.

