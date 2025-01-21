



if __name__=='__main__':
    wsi_paths = [] 
   
    for wsi_path in wsi_paths:
        slide_path = wsi_path
        print(slide_path)
        basename = os.path.basename(slide_path).split('.')[0]
        
        JSON_SAVE_PATH  = os.path.join('/content/drive/MyDrive/2024_Houston/02 - AIMAPathologyProject/Common/Superpixels/json_file', f'{basename}.json')
        os.makedirs(os.path.dirname(JSON_SAVE_PATH), exist_ok=True)
        print(os.listdir(os.path.dirname(JSON_SAVE_PATH)))

        ############################segment ############################################
        start  = time.time()

        print(slide_path)
        slide = openslide.open_slide(slide_path)

        (
            superpixel_labels,
            segmented_mask,
            downsample_factor,
            new_width,
            new_height,
            downscaled_region_array,
            lab_image )= superpixel_segmenting(slide, downsample_size = 1096, n_segments=500, compactness=10.0, start_label=0)

        ################################################################################

        print("len of super pixel:", len(superpixel_labels))
        print("new w, new h", new_width, new_height)
        print("downscaled_region_array:", downscaled_region_array.shape)

        print("processing time", time.time() - start)
        equalized_image = equalize_image(downscaled_region_array)



        foreground_superpixels, background_superpixels = identify_foreground_background(equalized_image, superpixel_labels)
        sp_plot = plot_foreground_boundaries_on_original_image(downscaled_region_array, superpixel_labels, foreground_superpixels)
        plt.figure(figsize=(20, 20))

        plt.subplot(1, 3, 1)  # (1 row, 3 columns, first subplot)
        plt.imshow(segmented_mask)
        plt.title("Segmented Mask")
        plt.axis('off')  # Turn off axis

        plt.subplot(1, 3, 2)  # (1 row, 3 columns, second subplot)
        plt.imshow(equalized_image)
        plt.title("Equalized Image")
        plt.axis('off')  # Turn off axis


        plt.subplot(1, 3, 3)
        plt.imshow(sp_plot)
        plt.title("Foreground Boundaries")
        plt.axis('off')  # Turn off axis

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5))
        plot_histogram(equalized_image)

        bounding_boxes, output_image_with_bboxes = get_bounding_boxes_for_foreground_segments(
            downscaled_region_array,
            superpixel_labels,
            foreground_superpixels
            )
        plt.figure(figsize=(12, 8))
        plt.imshow(output_image_with_bboxes)  # Display with matplotlib (expects RGB)
        plt.title("Foreground Superpixel Segments with Bounding Boxes")
        plt.axis('off')  # Turn off axis
        plt.show()

        ################################################################################
        # save the RESULT ----------------------------
        # Convert numpy arrays and scalars to Python-native types
        data_to_save = {
            'superpixel_labels': superpixel_labels.tolist(),  # Convert numpy array to list
            'downsample_factor': downsample_factor,  # Convert numpy scalar to Python float
            'new_width': new_width,  # Convert numpy int64 to Python int
            'new_height': new_height,  # Convert numpy int64 to Python int
            'downscaled_region_array': downscaled_region_array.tolist(),  # Convert numpy array to list
            'foreground_superpixels': [float(i) for i in foreground_superpixels],
            'background_superpixels': [float(i) for i in background_superpixels],
            'bounding_boxes': {str(k): list([float(j) for j in v]) for k, v in bounding_boxes.items()},
            'output_image_with_bboxes': output_image_with_bboxes.tolist(),   # Convert numpy array to list
            'superpixels_plot': sp_plot.tolist()
        }

        # Save the dictionary to a single JSON file
        with open(JSON_SAVE_PATH, 'w') as json_file:
            json.dump(data_to_save, json_file)

        # Print confirmation
        print("All results saved successfully in one JSON file!")
    