# Aula 5 - Feature Extraction e Matching

## Objetivo da Aula

Aprender técnicas avançadas de extração de características visuais e matching, explorando algoritmos como ORB, SIFT e aplicando-os em contextos reais como AR, biometria e inspeção industrial.

## Conteúdo Teórico

### Keypoints e Descritores

**Keypoints** são pontos específicos em uma imagem que possuem propriedades distintivas, como cantos, bordas ou regiões com alta variação de intensidade. São invariantes a transformações como rotação, escala e iluminação.

**Descritores** são vetores numéricos que codificam informações sobre a vizinhança de um keypoint, permitindo comparação entre diferentes keypoints.

### Algoritmos de Detecção de Keypoints

#### ORB (Oriented FAST and Rotated BRIEF)
- Combinação do detector FAST e descritor BRIEF
- Rápido e eficiente
- Livre de patentes
- Adequado para aplicações em tempo real

#### SIFT (Scale-Invariant Feature Transform)
- Invariante a escala e rotação
- Robusto a mudanças de iluminação
- Muito eficaz mas computacionalmente custoso
- Patenteado (expirou em 2020)

#### SURF (Speeded Up Robust Features)
- Versão mais rápida do SIFT
- Também invariante a escala e rotação
- Menos robusto que SIFT em algumas situações

### Matching de Características

O matching envolve encontrar correspondências entre keypoints de diferentes imagens, fundamental para:

- **Reconhecimento de objetos**
- **Reconstrução 3D**
- **Augmented Reality**
- **Biometria**
- **Inspeção de qualidade industrial**

## Atividade Prática

### Implementar Detector e Descritor ORB

```python
# src/features/orb_detector.py
import cv2
import numpy as np

class ORBDetector:
    def __init__(self, n_features=500, scale_factor=1.2, n_levels=8, edge_threshold=31, 
                 first_level=0, WTA_K=2, score_type=cv2.ORB_HARRIS_SCORE, patch_size=31, fast_threshold=20):
        """
        Inicializa o detector ORB
        """
        self.detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=WTA_K,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold
        )
    
    def detect_and_compute(self, image):
        """
        Detecta keypoints e computa descritores
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def draw_keypoints(self, image, keypoints, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        """
        Desenha keypoints na imagem
        """
        return cv2.drawKeypoints(image, keypoints, None, color=color, flags=flags)
    
    def match_features(self, descriptors1, descriptors2, matcher_type='bf', cross_check=True):
        """
        Realiza matching entre descritores
        """
        if matcher_type == 'bf':
            # Brute Force Matcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
            matches = matcher.match(descriptors1, descriptors2)
        elif matcher_type == 'flann':
            # FLANN Matcher (mais rápido para grandes conjuntos de dados)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            # Aplicar Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            matches = good_matches
        else:
            raise ValueError("Tipo de matcher não suportado")
        
        # Ordenar matches por distância
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def draw_matches(self, img1, kp1, img2, kp2, matches, n_matches=50):
        """
        Desenha matches entre duas imagens
        """
        # Limitar número de matches para visualização
        matches = matches[:min(n_matches, len(matches))]
        
        # Converter imagens para BGR se forem RGB
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        if len(img2.shape) == 3 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return result
```

### Implementar Detector SIFT (Alternativa)

```python
# src/features/sift_detector.py
import cv2
import numpy as np

class SIFTDetector:
    def __init__(self, n_features=4000, n_octave_layers=3, contrast_threshold=0.04, 
                 edge_threshold=10, sigma=1.6):
        """
        Inicializa o detector SIFT
        Nota: SIFT está disponível apenas em versões do OpenCV com licença não GPL
        """
        try:
            self.detector = cv2.SIFT_create(
                nfeatures=n_features,
                nOctaveLayers=n_octave_layers,
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold,
                sigma=sigma
            )
        except AttributeError:
            raise ImportError("SIFT não disponível nesta versão do OpenCV")
    
    def detect_and_compute(self, image):
        """
        Detecta keypoints e computa descritores SIFT
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def draw_keypoints(self, image, keypoints, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        """
        Desenha keypoints na imagem
        """
        return cv2.drawKeypoints(image, keypoints, None, color=color, flags=flags)
    
    def match_features(self, descriptors1, descriptors2, matcher_type='bf', cross_check=False):
        """
        Realiza matching entre descritores SIFT
        """
        if matcher_type == 'bf':
            # Brute Force com norma L2 (adequada para SIFT)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            matches = matcher.match(descriptors1, descriptors2)
        elif matcher_type == 'flann':
            # FLANN para SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            # Aplicar Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            matches = good_matches
        else:
            raise ValueError("Tipo de matcher não suportado")
        
        # Ordenar matches por distância
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
```

### Implementar Comparação de Imagens

```python
# src/features/image_matching.py
import cv2
import numpy as np
from .orb_detector import ORBDetector

class ImageMatcher:
    def __init__(self, detector_type='orb'):
        if detector_type == 'orb':
            self.detector = ORBDetector()
        else:
            raise ValueError("Tipo de detector não suportado")
    
    def compare_two_images(self, img1, img2, min_matches=10):
        """
        Compara duas imagens e retorna número de matches e similaridade
        """
        # Detectar e computar keypoints e descritores
        kp1, desc1 = self.detector.detect_and_compute(img1)
        kp2, desc2 = self.detector.detect_and_compute(img2)
        
        if desc1 is None or desc2 is None:
            return {
                'matches_count': 0,
                'similarity': 0.0,
                'has_match': False,
                'keypoints1': len(kp1) if kp1 else 0,
                'keypoints2': len(kp2) if kp2 else 0
            }
        
        # Realizar matching
        matches = self.detector.match_features(desc1, desc2)
        
        # Calcular similaridade baseada em número de matches
        similarity = min(len(matches) / min(len(kp1), len(kp2)), 1.0) if kp1 and kp2 else 0.0
        
        result = {
            'matches_count': len(matches),
            'similarity': similarity,
            'has_match': len(matches) >= min_matches,
            'keypoints1': len(kp1),
            'keypoints2': len(kp2),
            'matches': matches,
            'keypoints1_raw': kp1,
            'keypoints2_raw': kp2
        }
        
        return result
    
    def find_template_in_image(self, template, image, threshold=0.7):
        """
        Encontra uma template em uma imagem maior usando feature matching
        """
        # Detectar features na template e na imagem
        kp_template, desc_template = self.detector.detect_and_compute(template)
        kp_image, desc_image = self.detector.detect_and_compute(image)
        
        if desc_template is None or desc_image is None:
            return []
        
        # Matching
        matches = self.detector.match_features(desc_template, desc_image)
        
        # Filtrar matches baseado em threshold
        filtered_matches = [m for m in matches if m.distance < threshold * 100]
        
        if len(filtered_matches) < 10:  # Mínimo de matches para considerar uma correspondência
            return []
        
        # Extrair coordenadas dos keypoints correspondentes
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_image[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        
        # Calcular homografia
        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is not None:
                # Obter cantos da template
                h, w = template.shape[:2]
                template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                # Transformar cantos para a imagem
                image_corners = cv2.perspectiveTransform(template_corners, homography)
                # Converter para formato adequado
                image_corners = np.int32(image_corners).reshape(-1, 2)
                return image_corners
        except:
            pass
        
        return []
    
    def calculate_feature_similarity(self, img1, img2, metric='ratio'):
        """
        Calcula similaridade entre duas imagens baseado em features
        """
        comparison = self.compare_two_images(img1, img2)
        
        if metric == 'ratio':
            # Similaridade baseada na razão de matches
            return comparison['similarity']
        elif metric == 'count':
            # Similaridade baseada no número absoluto de matches
            return min(comparison['matches_count'] / 100, 1.0)  # Normalizar
        else:
            raise ValueError("Métrica não suportada")
```

### Implementar Aplicações Reais

```python
# src/features/applications.py
import cv2
import numpy as np
from .image_matching import ImageMatcher

class FeatureApplications:
    def __init__(self):
        self.matcher = ImageMatcher()
    
    def augmented_reality_overlay(self, reference_img, live_frame):
        """
        Demonstração de overlay AR baseado em matching de features
        """
        # Encontrar correspondências entre referência e frame ao vivo
        comparison = self.matcher.compare_two_images(reference_img, live_frame)
        
        if comparison['has_match']:
            matches = comparison['matches']
            kp_ref = comparison['keypoints1_raw']
            kp_live = comparison['keypoints2_raw']
            
            # Extrair pontos correspondentes
            if len(matches) >= 4:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_live[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Calcular homografia
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if homography is not None:
                    # Definir região para overlay (ex: cantos da referência)
                    h, w = reference_img.shape[:2]
                    ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    live_corners = cv2.perspectiveTransform(ref_corners, homography)
                    
                    # Retornar cantos para desenho de overlay
                    return np.int32(live_corners).reshape(-1, 2)
        
        return None
    
    def quality_inspection(self, reference_img, test_img, threshold=0.7):
        """
        Sistema de inspeção de qualidade baseado em matching
        """
        similarity = self.matcher.calculate_feature_similarity(reference_img, test_img)
        
        result = {
            'similarity': similarity,
            'pass': similarity >= threshold,
            'difference': abs(similarity - threshold)
        }
        
        return result
    
    def object_recognition(self, object_templates, scene_img, threshold=0.6):
        """
        Sistema de reconhecimento de objetos em cena
        """
        recognized_objects = []
        
        for obj_name, template in object_templates.items():
            # Tentar encontrar o objeto na cena
            corners = self.matcher.find_template_in_image(template, scene_img)
            
            if len(corners) > 0:
                # Calcular confiança baseada em número de matches
                comparison = self.matcher.compare_two_images(template, scene_img)
                confidence = min(1.0, comparison['matches_count'] / 50)  # Normalizar
                
                if confidence >= threshold:
                    recognized_objects.append({
                        'name': obj_name,
                        'location': corners,
                        'confidence': confidence
                    })
        
        return recognized_objects
```

### Exemplo de Uso Integrado

```python
# src/examples/feature_extraction_example.py
from features.orb_detector import ORBDetector
from features.image_matching import ImageMatcher
from features.applications import FeatureApplications
from utils.io import load_image_rgb, show_image
import matplotlib.pyplot as plt

def demonstrate_feature_extraction():
    """Demonstra extração e matching de features"""
    # Carregar imagens
    img1 = load_image_rgb("data/raw/exemplo1.jpg")  # Substitua pelos caminhos reais
    img2 = load_image_rgb("data/raw/exemplo2.jpg")  # Imagem similar ou transformada
    
    # Inicializar detector
    detector = ORBDetector(n_features=500)
    
    # Detectar keypoints e descritores
    kp1, desc1 = detector.detect_and_compute(img1)
    kp2, desc2 = detector.detect_and_compute(img2)
    
    print(f"Keypoints na imagem 1: {len(kp1) if kp1 else 0}")
    print(f"Keypoints na imagem 2: {len(kp2) if kp2 else 0}")
    
    # Desenhar keypoints
    img1_kp = detector.draw_keypoints(img1, kp1)
    img2_kp = detector.draw_keypoints(img2, kp2)
    
    # Realizar matching
    matches = detector.match_features(desc1, desc2)
    print(f"Matches encontrados: {len(matches)}")
    
    # Desenhar matches
    if matches:
        img_matches = detector.draw_matches(img1, kp1, img2, kp2, matches, n_matches=20)
    
    # Comparar imagens
    matcher = ImageMatcher()
    comparison = matcher.compare_two_images(img1, img2)
    print(f"Similaridade: {comparison['similarity']:.3f}")
    print(f"Tem correspondência: {comparison['has_match']}")
    
    # Visualizar resultados
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].imshow(img1_kp)
    axes[0,0].set_title(f'Keypoints - Imagem 1 ({len(kp1) if kp1 else 0})')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img2_kp)
    axes[0,1].set_title(f'Keypoints - Imagem 2 ({len(kp2) if kp2 else 0})')
    axes[0,1].axis('off')
    
    if matches and 'img_matches' in locals():
        axes[1,0].imshow(img_matches)
        axes[1,0].set_title(f'Matches ({len(matches)} encontrados)')
        axes[1,0].axis('off')
    else:
        axes[1,0].text(0.5, 0.5, 'Nenhum match encontrado', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1,0].transAxes, fontsize=14)
        axes[1,0].axis('off')
    
    # Aplicação: inspeção de qualidade
    app = FeatureApplications()
    if img1 is not None and img2 is not None:
        inspection_result = app.quality_inspection(img1, img2)
        axes[1,1].text(0.1, 0.8, f'Similaridade: {inspection_result["similarity"]:.3f}', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].text(0.1, 0.7, f'Resultado: {"PASS" if inspection_result["pass"] else "FAIL"}', 
                      transform=axes[1,1].transAxes, fontsize=12, 
                      color='green' if inspection_result["pass"] else 'red')
        axes[1,1].set_title('Inspeção de Qualidade')
    else:
        axes[1,1].text(0.5, 0.5, 'Imagens não carregadas', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1,1].transAxes, fontsize=14)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

def demonstrate_object_recognition():
    """Demonstra reconhecimento de objetos"""
    # Carregar cena e templates
    scene = load_image_rgb("data/raw/cena.jpg")  # Cena com múltiplos objetos
    template1 = load_image_rgb("data/raw/template1.jpg")  # Template de objeto 1
    template2 = load_image_rgb("data/raw/template2.jpg")  # Template de objeto 2
    
    # Criar dicionário de templates
    templates = {
        'objeto1': template1,
        'objeto2': template2
    }
    
    # Inicializar aplicação
    app = FeatureApplications()
    
    # Reconhecer objetos
    recognized = app.object_recognition(templates, scene)
    
    print(f"Objetos reconhecidos: {len(recognized)}")
    for obj in recognized:
        print(f"  - {obj['name']}: confiança {obj['confidence']:.3f}")
    
    # Desenhar resultados na cena
    result_scene = scene.copy()
    for obj in recognized:
        points = obj['location']
        # Desenhar polígono ao redor do objeto reconhecido
        cv2.polylines(result_scene, [points], True, (0, 255, 0), 2)
        # Adicionar texto
        cv2.putText(result_scene, f"{obj['name']} ({obj['confidence']:.2f})", 
                   tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Mostrar resultado
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(scene)
    plt.title('Cena Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_scene)
    plt.title('Objetos Reconhecidos')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Executar demonstrações
    try:
        demonstrate_feature_extraction()
    except Exception as e:
        print(f"Erro na demonstração de extração de features: {e}")
    
    try:
        demonstrate_object_recognition()
    except Exception as e:
        print(f"Erro na demonstração de reconhecimento de objetos: {e}")
```

## Resultado Esperado

Nesta aula, você:

1. Aprendeu sobre keypoints e descritores de imagens
2. Implementou o detector e descritor ORB
3. Criou funcionalidades para matching de features
4. Desenvolveu aplicações práticas como AR, inspeção de qualidade e reconhecimento de objetos
5. Testou os algoritmos em diferentes cenários
6. Entendeu as aplicações reais dessas técnicas em AR, biometria e inspeção industrial

Essas técnicas são fundamentais para muitas aplicações avançadas de visão computacional e formam a base para sistemas mais complexos que utilizam aprendizado de máquina.