library(shiny)
library(ggplot2)
library(dplyr)
library(tools)
library(stringr)
library(scales)

# Helper to format file names
format_label <- function(filename) {
  name <- file_path_sans_ext(filename)
  parts <- unlist(strsplit(name, "_"))
  parts <- tolower(parts)
  parts <- tools::toTitleCase(parts)
  parts[1] <- toupper(parts[1])
  label <- paste(parts, collapse = " ")
  return(label)
}

# Load accuracy data
load_accuracy_tables <- function() {
  accuracy_dir <- file.path("..", "results", "accuracy_tables")
  files <- list.files(accuracy_dir, pattern = "\\.csv$", full.names = TRUE)
  
  df_list <- lapply(files, function(file) {
    df <- read.csv(file)
    df$source_file <- basename(file)
    return(df)
  })
  
  combined_df <- bind_rows(df_list)
  return(combined_df)
}

accuracy_data <- load_accuracy_tables()

# Create label map for select input
file_choices <- unique(accuracy_data$source_file)
file_labels <- sapply(file_choices, format_label)
names(file_choices) <- file_labels

# UI
ui <- fluidPage(
  titlePanel("Accuracy Results of Experiments"),
  
  div(style = "margin-bottom: 20px;",
      selectInput("file_select", "Select File:", choices = file_choices, width = "300px")
  ),
  
  plotOutput("accuracyPlot", height = "450px")  # Reduced from 600px
)

# Server
server <- function(input, output) {
  
  render_accuracy_plot <- reactive({
    plot_data <- accuracy_data %>%
      filter(source_file == input$file_select) %>%
      mutate(method = factor(method, levels = unique(method)))
    
    ggplot(plot_data, aes(x = method, y = accuracy * 100)) +
      geom_bar(stat = "identity", width = 0.7, fill = "#DDDDDD", color = "black") +
      geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)),
                vjust = -0.5, size = 5) +
      scale_y_continuous(
        labels = percent_format(scale = 1),
        breaks = seq(0, 100, by = 10),
        limits = c(0, 100),
        expand = c(0, 0)
      ) +
      labs(
        title = format_label(input$file_select),
        subtitle = NULL,
        x = "Method",
        y = "Accuracy"
      ) +
      theme_classic(base_size = 14, base_family = "Helvetica") +
      theme(
        axis.text.x = element_text(angle = 0, hjust = 0.5),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        legend.position = "none"
      )
  })
  
  output$accuracyPlot <- renderPlot({
    render_accuracy_plot()
  })
}

# Run the app
shinyApp(ui = ui, server = server)
