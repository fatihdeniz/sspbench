import os
import streamlit as st
import yaml
import json
import base64

class AnnotationApp:
    def __init__(self):
        self.config = self.load_config()
        self.session_state_initialization()

    def session_state_initialization(self):
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'data' not in st.session_state:
            st.session_state.data = []
        if 'selected_keys' not in st.session_state:
            st.session_state.selected_keys = []
        if 'annotation_filepath' not in st.session_state:
            st.session_state.annotation_filepath = ""
        if 'dataset_name' not in st.session_state:
            st.session_state.dataset_name = ""

    def load_css(self):
        with open("trusteval/src/annotation/annotation.css") as css_file:
            css_content = css_file.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

    def load_config(self):
        with open("trusteval/src/config/annotation_config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config

    def initialize_annotation_state(self, index):
        for list_name in self.config:
            if f"{list_name}_{index}" not in st.session_state:
                st.session_state[f"{list_name}_{index}"] = None  # No default value
            if f"feedback_{index}" not in st.session_state:
                st.session_state[f"feedback_{index}"] = ""

    def load_data_file(self, uploaded_file):
        dataset_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        annotation_filename = f"{dataset_name}_annotation.json"
        annotation_filepath = os.path.join('data', dataset_name, annotation_filename)
        os.makedirs(os.path.dirname(annotation_filepath), exist_ok=True)

        if os.path.exists(annotation_filepath):
            with open(annotation_filepath, "r") as file:
                data = json.load(file)
                st.success(f"Loaded annotation file: {annotation_filename}")
        else:
            data = json.load(uploaded_file)
            for item in data:
                for list_name in self.config:
                    item[list_name] = ""
                item['feedback'] = ""
            with open(annotation_filepath, "w") as file:
                json.dump(data, file, indent=4)
            st.success(f"Created new annotation file: {annotation_filename}")

        return data, annotation_filepath, dataset_name

    def save_annotations(self, data, filepath):
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

    def on_index_change(self):
        st.session_state.current_index = st.session_state.item_index

    def go_previous(self):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1

    def go_next(self):
        if st.session_state.current_index < len(st.session_state.data) - 1:
            st.session_state.current_index += 1

    def display_annotation_interface(self, data, selected_keys, current_index, show_image=False):
        self.initialize_annotation_state(current_index)
        item = data[current_index]
        st.subheader("Item Data:")
        st.write({key: item.get(key, None) for key in selected_keys})

        def display_images_in_grid(item, image_keys=None, image_paths=None, columns_per_row=3):
            """
            Display images in a 3-column grid layout.
            
            Args:
                item: The current item from the dataset containing image paths (not used for image_paths list).
                image_keys: List of keys containing the image paths in `item` (for when keys are provided).
                image_paths: List of image paths to display (for when direct paths are provided).
                columns_per_row: Number of columns to display per row (default is 3).
            """
            if image_keys:
                # Collect the image paths from the item using image_keys
                image_paths = []
                for image_key in image_keys:
                    if image_key in item:
                        image_path = item[image_key]
                        image_paths.append((image_key, image_path))  # Store both key and path
            elif image_paths is not None:
                # When image_paths is provided directly, no need to collect from item
                image_paths = [(str(idx+1), path) for idx, path in enumerate(image_paths)]  # Add numbered captions (1, 2, 3...)
            else:
                st.warning("No image data provided!")
                return

            # Determine how many rows of images are needed
            num_images = len(image_paths)
            num_rows = (num_images + columns_per_row - 1) // columns_per_row  # Ceiling division to calculate rows
            
            # Display images in the specified column grid layout
            for row_idx in range(num_rows):
                # Create a row with specified columns
                cols = st.columns(columns_per_row)
                
                # Determine which images belong in this row
                start_idx = row_idx * columns_per_row
                end_idx = min(start_idx + columns_per_row, num_images)
                
                # Display images in the columns for this row
                for idx in range(start_idx, end_idx):
                    image_caption, image_path = image_paths[idx]
                    if os.path.exists(image_path):
                        with cols[idx - start_idx]:  # Use the appropriate column in the row
                            st.image(image_path, caption=image_caption, use_container_width=True)
                    else:
                        with cols[idx - start_idx]:
                            st.warning(f"Image not found: {image_path}")


        def display_options_in_grid(data, current_index, columns_per_row=3):
            """
            Display options in a 3-column grid layout based on the configuration.
            
            Args:
                self: The instance of the class containing the configuration.
                data: The data being annotated.
                current_index: The index of the current item being annotated.
                columns_per_row: The number of columns to display per row (default is 3).
            """
            total_columns = len(self.config)

            # Calculate the number of rows needed
            num_rows = (total_columns + columns_per_row - 1) // columns_per_row  # Ceiling division

            # Create a list of columns with the specified number of columns per row
            for row_idx in range(num_rows):
                # Create columns for this row, adjusting for the number of remaining columns
                cols = st.columns(columns_per_row)
                
                # Determine the start and end indices for the current row
                start_idx = row_idx * columns_per_row
                end_idx = min(start_idx + columns_per_row, total_columns)
                
                for idx in range(start_idx, end_idx):
                    list_name, options = list(self.config.items())[idx]
                    with cols[idx - start_idx]:  # Use the column for the current element
                        def update_choice(list_name=list_name):
                            st.session_state[f"{list_name}_{current_index}"] = st.session_state[f"{list_name}_radio_{current_index}"]
                            data[current_index][list_name] = st.session_state[f"{list_name}_radio_{current_index}"]
                            self.save_annotations(data, st.session_state.annotation_filepath)

                        # Title and radio button inside the current column
                        st.markdown(f"#### {list_name}")
                        st.radio(
                            list_name,
                            options,
                            index=None,
                            key=f"{list_name}_radio_{current_index}",
                            on_change=update_choice,
                            label_visibility="collapsed",
                        )                 
        image_urls = item['image_urls']
                                                
        if show_image:

            display_images_in_grid(item,image_urls = item['image_urls'])
            display_options_in_grid(data, current_index)
            
            

        feedback_col, status_col = st.columns(2)
        with feedback_col:
            def update_feedback():
                st.session_state[f"feedback_{current_index}"] = st.session_state[f"feedback_textarea_{current_index}"]
                data[current_index]['feedback'] = st.session_state[f"feedback_{current_index}"]
                self.save_annotations(data, st.session_state.annotation_filepath)

            st.subheader(f"Feedback:")
            st.text_area(
                f"feedback",
                value=st.session_state[f"feedback_{current_index}"],
                key=f"feedback_textarea_{current_index}",
                on_change=update_feedback,
                label_visibility="collapsed",
            )

        with status_col:
            st.subheader("Status")
            status = {list_name: "Annotated" if item.get(list_name) != "" else "Not Annotated" for list_name in self.config}
            st.write(status)

        prev_col, next_col = st.columns([1, 1])
        with prev_col:
            st.button("Previous", on_click=self.go_previous)
        with next_col:
            st.button("Next", on_click=self.go_next)


    def display_overall_status(self, data):
        overall_status = {}
        total_items = len(data)

        for list_name, options in self.config.items():
            option_counts = {option: 0 for option in options}
            option_counts["annotated_items"] = 0
            for item in data:
                selected_option = item.get(list_name, None)
                if selected_option in option_counts:
                    option_counts[selected_option] += 1
                if selected_option != "":
                    option_counts["annotated_items"] += 1
            annotated_items = option_counts["annotated_items"]
            option_percentages = {option: f"{(count / annotated_items * 100):.3f}%" if total_items > 0 else "0.000%"
                                for option, count in option_counts.items() if option != "annotated_items" }

            overall_status[list_name] = {
                "Options": option_counts,
                "Percentages": option_percentages,
            }

        st.sidebar.subheader("Overall Status of Annotations:")
        st.sidebar.markdown("---")

        for list_name, status in overall_status.items():
            st.sidebar.markdown(f"### Key: {list_name}")
            st.sidebar.write("Options Count:", status["Options"])
            st.sidebar.write("Options Percentage:", status["Percentages"])
            st.sidebar.markdown("---")


    def run(self):
        self.load_css()
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Go to", ["Configuration", "Text Annotation Platform", "Image Annotation Platform"], label_visibility="collapsed")

        if page == "Configuration":
            st.title("Configuration Page")
            uploaded_file = st.file_uploader("Upload a JSON data file", type="json")
            if uploaded_file is not None:
                data, annotation_filepath, dataset_name = self.load_data_file(uploaded_file)
                st.session_state.annotation_filepath = annotation_filepath
                st.session_state.dataset_name = dataset_name
                st.write("JSON Data:", data)

                all_keys = set()
                for item in data:
                    all_keys.update(item.keys())

                filtered_keys = [key for key in all_keys if key not in self.config and key != 'feedback']
                selected_keys = st.multiselect("Select keys to display", filtered_keys)
                if st.button("Load Selected Keys"):
                    st.session_state.selected_keys = selected_keys
                    st.session_state.data = data
                    st.session_state.current_index = 0
                    for i in range(len(data)):
                        self.initialize_annotation_state(i)

        elif page == "Text Annotation Platform":
            st.title("Text Annotation Platform")
            if st.session_state.selected_keys and st.session_state.data:
                selected_keys = st.session_state.selected_keys
                data = st.session_state.data
                st.sidebar.number_input(
                    "Select Item Index",
                    min_value=0,
                    max_value=len(data) - 1,
                    value=st.session_state.current_index,
                    step=1,
                    key="item_index",
                    on_change=self.on_index_change
                )
                st.sidebar.markdown(f'Total items: {len(data)}')
                current_index = st.session_state.current_index
                self.display_annotation_interface(data, selected_keys, current_index, show_image=False)

                if st.sidebar.button("Show Status"):
                    self.display_overall_status(data)

        elif page == "Image Annotation Platform":
            st.title("Image Annotation Platform")
            if st.session_state.selected_keys and st.session_state.data:
                selected_keys = st.session_state.selected_keys
                data = st.session_state.data
                st.sidebar.number_input(
                    "Select Item Index",
                    min_value=0,
                    max_value=len(data) - 1,
                    value=st.session_state.current_index,
                    step=1,
                    key="item_index",
                    on_change=self.on_index_change
                )
                st.sidebar.markdown(f'Total items: {len(data)}')
                current_index = st.session_state.current_index
                self.display_annotation_interface(data, selected_keys, current_index, show_image=True)

                if st.sidebar.button("Show Status"):
                    self.display_overall_status(data)

if __name__ == "__main__":
    app = AnnotationApp()
    app.run()
